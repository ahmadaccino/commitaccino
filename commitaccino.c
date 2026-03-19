/*
 * commitaccino - AI-powered git commits via AWS Bedrock
 *
 * Reads git diff, sends to a Bedrock model, generates a commit message,
 * stages everything, and commits. Lightning fast.
 *
 * Requires: libcurl, AWS_BEARER_TOKEN_BEDROCK, AWS_REGION env vars
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <curl/curl.h>

/* ── Constants ──────────────────────────────────────────────────────── */

#define MAX_DIFF_SIZE    (100 * 1024)
#define MAX_MODEL_ID     256
#define NUM_MODELS       5
#define VERSION          "1.0.0"

/* ── ANSI Colors ────────────────────────────────────────────────────── */

#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_RED     "\033[31m"
#define C_GREEN   "\033[32m"
#define C_YELLOW  "\033[33m"
#define C_CYAN    "\033[36m"

/* ── Preset Models ──────────────────────────────────────────────────── */

static const char *MODELS[NUM_MODELS] = {
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.amazon.nova-micro-v1:0",
    "us.amazon.nova-lite-v1:0",
    "us.meta.llama4-scout-17b-instruct-v1:0"
};

static const char *MODEL_NAMES[NUM_MODELS] = {
    "Claude Sonnet 4",
    "Claude Haiku 4",
    "Amazon Nova Micro",
    "Amazon Nova Lite",
    "Llama 4 Scout 17B"
};

/* ── Timer ──────────────────────────────────────────────────────────── */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ── Run Shell Command ──────────────────────────────────────────────── */

static char *run_cmd(const char *cmd) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    size_t cap = 4096, len = 0;
    char *buf = malloc(cap);
    if (!buf) { pclose(fp); return NULL; }

    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, fp)) > 0) {
        len += n;
        if (len + 1 >= cap) {
            cap *= 2;
            char *tmp = realloc(buf, cap);
            if (!tmp) { free(buf); pclose(fp); return NULL; }
            buf = tmp;
        }
    }
    buf[len] = '\0';
    pclose(fp);
    return buf;
}

static int run_cmd_status(const char *cmd) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return -1;
    char tmp[256];
    while (fread(tmp, 1, sizeof(tmp), fp) > 0) {}
    int status = pclose(fp);
    return WEXITSTATUS(status);
}

/* ── JSON Escape ────────────────────────────────────────────────────── */

static char *json_escape(const char *src, size_t src_len) {
    size_t cap = src_len * 6 + 1;
    char *dst = malloc(cap);
    if (!dst) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < src_len; i++) {
        unsigned char c = (unsigned char)src[i];
        switch (c) {
            case '"':  dst[j++] = '\\'; dst[j++] = '"';  break;
            case '\\': dst[j++] = '\\'; dst[j++] = '\\'; break;
            case '\n': dst[j++] = '\\'; dst[j++] = 'n';  break;
            case '\r': dst[j++] = '\\'; dst[j++] = 'r';  break;
            case '\t': dst[j++] = '\\'; dst[j++] = 't';  break;
            default:
                if (c < 0x20) {
                    j += (size_t)snprintf(dst + j, 7, "\\u%04x", c);
                } else {
                    dst[j++] = (char)c;
                }
                break;
        }
    }
    dst[j] = '\0';
    return dst;
}

/* ── JSON Parsing ───────────────────────────────────────────────────── */

static char *extract_json_string_after(const char *json, const char *anchor, const char *key) {
    const char *p = json;
    if (anchor) {
        p = strstr(json, anchor);
        if (!p) return NULL;
    }

    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    p = strstr(p, pattern);
    if (!p) return NULL;
    p += strlen(pattern);

    while (*p && (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    if (*p != '"') return NULL;
    p++;

    size_t cap = 2048, len = 0;
    char *val = malloc(cap);
    if (!val) return NULL;

    while (*p && *p != '"') {
        if (len + 2 >= cap) {
            cap *= 2;
            char *tmp = realloc(val, cap);
            if (!tmp) { free(val); return NULL; }
            val = tmp;
        }
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case '"':  val[len++] = '"';  break;
                case '\\': val[len++] = '\\'; break;
                case 'n':  val[len++] = '\n'; break;
                case 'r':  val[len++] = '\r'; break;
                case 't':  val[len++] = '\t'; break;
                case '/':  val[len++] = '/';  break;
                default:   val[len++] = '\\'; val[len++] = *p; break;
            }
        } else {
            val[len++] = *p;
        }
        p++;
    }
    val[len] = '\0';
    return val;
}

static long json_extract_long(const char *json, const char *key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char *p = strstr(json, pattern);
    if (!p) return -1;
    p += strlen(pattern);

    while (*p && (*p == ' ' || *p == ':' || *p == '\t')) p++;
    return strtol(p, NULL, 10);
}

/* ── Config File ────────────────────────────────────────────────────── */

static void get_config_path(char *path, size_t size) {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(path, size, "%s/.commitaccino", home);
}

static int load_config(char *model_id, size_t max_len) {
    char path[512];
    get_config_path(path, sizeof(path));

    FILE *fp = fopen(path, "r");
    if (!fp) return 0;

    if (!fgets(model_id, (int)max_len, fp)) {
        fclose(fp);
        return 0;
    }
    fclose(fp);

    size_t len = strlen(model_id);
    while (len > 0 && (model_id[len - 1] == '\n' || model_id[len - 1] == '\r'))
        model_id[--len] = '\0';

    return len > 0;
}

static int save_config(const char *model_id) {
    char path[512];
    get_config_path(path, sizeof(path));

    FILE *fp = fopen(path, "w");
    if (!fp) return 0;
    fprintf(fp, "%s\n", model_id);
    fclose(fp);
    return 1;
}

static void prompt_model_selection(char *model_id, size_t max_len) {
    printf("\n  %s%scommitaccino%s %s- pick your default model%s\n\n",
           C_BOLD, C_CYAN, C_RESET, C_DIM, C_RESET);

    for (int i = 0; i < NUM_MODELS; i++) {
        printf("    %s%d%s) %s%-22s %s%s%s\n",
               C_BOLD, i + 1, C_RESET,
               C_CYAN, MODEL_NAMES[i],
               C_DIM, MODELS[i], C_RESET);
    }

    printf("\n  Choice [1-%d]: ", NUM_MODELS);
    fflush(stdout);

    char input[32];
    if (!fgets(input, sizeof(input), stdin)) {
        fprintf(stderr, "  %sError reading input%s\n", C_RED, C_RESET);
        exit(1);
    }

    int choice = atoi(input);
    if (choice < 1 || choice > NUM_MODELS) {
        fprintf(stderr, "  %sInvalid choice, defaulting to 1.%s\n", C_YELLOW, C_RESET);
        choice = 1;
    }

    snprintf(model_id, max_len, "%s", MODELS[choice - 1]);
    save_config(model_id);
    printf("\n  %s✓ Saved: %s%s\n\n", C_GREEN, model_id, C_RESET);
}

/* ── Diff Stats ─────────────────────────────────────────────────────── */

static void parse_diff_stats(const char *stat_output, int *files, int *ins, int *del) {
    *files = *ins = *del = 0;

    /* Find the last line which has the summary */
    const char *last_line = stat_output;
    const char *p = stat_output;
    while (*p) {
        if (*p == '\n' && *(p + 1))
            last_line = p + 1;
        p++;
    }

    sscanf(last_line, " %d file", files);

    const char *ins_p = strstr(last_line, "insertion");
    if (ins_p) {
        const char *n = ins_p - 1;
        while (n > last_line && *n == ' ') n--;
        while (n > last_line && *(n - 1) >= '0' && *(n - 1) <= '9') n--;
        *ins = atoi(n);
    }

    const char *del_p = strstr(last_line, "deletion");
    if (del_p) {
        const char *n = del_p - 1;
        while (n > last_line && *n == ' ') n--;
        while (n > last_line && *(n - 1) >= '0' && *(n - 1) <= '9') n--;
        *del = atoi(n);
    }
}

/* ── curl Write Callback ───────────────────────────────────────────── */

struct response_buf {
    char  *data;
    size_t size;
    size_t cap;
};

static size_t write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t total = size * nmemb;
    struct response_buf *rb = (struct response_buf *)userp;

    while (rb->size + total + 1 > rb->cap) {
        rb->cap *= 2;
        char *tmp = realloc(rb->data, rb->cap);
        if (!tmp) return 0;
        rb->data = tmp;
    }

    memcpy(rb->data + rb->size, contents, total);
    rb->size += total;
    rb->data[rb->size] = '\0';
    return total;
}

/* ── URL Encode Model ID ───────────────────────────────────────────── */

static void url_encode_model(const char *model, char *out, size_t out_size) {
    size_t j = 0;
    for (size_t i = 0; model[i] && j + 3 < out_size; i++) {
        if (model[i] == ':') {
            out[j++] = '%';
            out[j++] = '3';
            out[j++] = 'A';
        } else {
            out[j++] = model[i];
        }
    }
    out[j] = '\0';
}

/* ── Bedrock Converse API ───────────────────────────────────────────── */

static char *call_bedrock(const char *region, const char *token,
                          const char *model_id, const char *diff,
                          long *in_tokens, long *out_tokens) {
    *in_tokens = -1;
    *out_tokens = -1;

    CURL *curl = curl_easy_init();
    if (!curl) return NULL;

    /* Build URL */
    char encoded_model[512];
    url_encode_model(model_id, encoded_model, sizeof(encoded_model));

    char url[1024];
    snprintf(url, sizeof(url),
             "https://bedrock-runtime.%s.amazonaws.com/model/%s/converse",
             region, encoded_model);

    /* Escape diff for JSON */
    size_t diff_len = strlen(diff);
    char *escaped_diff = json_escape(diff, diff_len);
    if (!escaped_diff) { curl_easy_cleanup(curl); return NULL; }

    /* System prompt */
    static const char *sys_prompt =
        "You are a git commit message generator. Given a git diff, produce ONLY "
        "a concise commit message. No explanation, no markdown, no quotes, no "
        "code fences. Just the raw commit message text. Use conventional commit "
        "format when appropriate (feat:, fix:, refactor:, docs:, chore:, style:, "
        "test:). Keep the subject line under 72 characters. Add a blank line and "
        "bullet points for details only if the change is complex.";

    char *escaped_sys = json_escape(sys_prompt, strlen(sys_prompt));
    if (!escaped_sys) { free(escaped_diff); curl_easy_cleanup(curl); return NULL; }

    /* Build request JSON */
    size_t body_cap = strlen(escaped_diff) + strlen(escaped_sys) + 4096;
    char *body = malloc(body_cap);
    if (!body) { free(escaped_diff); free(escaped_sys); curl_easy_cleanup(curl); return NULL; }

    snprintf(body, body_cap,
        "{\"system\":[{\"text\":\"%s\"}],"
        "\"messages\":[{\"role\":\"user\",\"content\":[{\"text\":"
        "\"Generate a commit message for this diff:\\n\\n%s\"}]}],"
        "\"inferenceConfig\":{\"maxTokens\":300,\"temperature\":0.3}}",
        escaped_sys, escaped_diff);

    free(escaped_diff);
    free(escaped_sys);

    /* Response buffer */
    struct response_buf rb = { .data = malloc(4096), .size = 0, .cap = 4096 };
    if (!rb.data) { free(body); curl_easy_cleanup(curl); return NULL; }
    rb.data[0] = '\0';

    /* Auth header */
    char auth_header[4096];
    snprintf(auth_header, sizeof(auth_header), "Authorization: Bearer %s", token);

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");
    headers = curl_slist_append(headers, auth_header);

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(body));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &rb);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TCP_FASTOPEN, 1L);

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    free(body);

    if (res != CURLE_OK) {
        fprintf(stderr, "  %s✗ curl error: %s%s\n", C_RED, curl_easy_strerror(res), C_RESET);
        free(rb.data);
        return NULL;
    }

    if (http_code != 200) {
        fprintf(stderr, "  %s✗ API error (HTTP %ld)%s\n", C_RED, http_code, C_RESET);
        if (http_code == 403 || http_code == 401) {
            fprintf(stderr, "  %sCheck your AWS_BEARER_TOKEN_BEDROCK - it may be expired or invalid%s\n",
                    C_DIM, C_RESET);
        } else if (http_code == 400) {
            fprintf(stderr, "  %sBad request - the model may not support the Converse API%s\n",
                    C_DIM, C_RESET);
        } else if (http_code == 404) {
            fprintf(stderr, "  %sModel not found - check model ID and region%s\n",
                    C_DIM, C_RESET);
        }
        fprintf(stderr, "  %sResponse: %.500s%s\n", C_DIM, rb.data, C_RESET);
        free(rb.data);
        return NULL;
    }

    /* Parse response - find text inside content array */
    char *message = extract_json_string_after(rb.data, "\"content\"", "text");
    *in_tokens = json_extract_long(rb.data, "inputTokens");
    *out_tokens = json_extract_long(rb.data, "outputTokens");

    free(rb.data);
    return message;
}

/* ── Strip / Clean Message ──────────────────────────────────────────── */

static void strip_message(char *msg) {
    if (!msg) return;

    /* Strip leading whitespace */
    char *start = msg;
    while (*start && (*start == ' ' || *start == '\n' || *start == '\r' || *start == '\t'))
        start++;
    if (start != msg)
        memmove(msg, start, strlen(start) + 1);

    /* Strip surrounding double quotes */
    size_t len = strlen(msg);
    if (len >= 2 && msg[0] == '"' && msg[len - 1] == '"') {
        memmove(msg, msg + 1, len - 2);
        msg[len - 2] = '\0';
        len -= 2;
    }

    /* Strip markdown code fences */
    if (strncmp(msg, "```", 3) == 0) {
        char *nl = strchr(msg, '\n');
        if (nl) {
            memmove(msg, nl + 1, strlen(nl + 1) + 1);
        }
        len = strlen(msg);
        if (len >= 3 && strcmp(msg + len - 3, "```") == 0)
            msg[len - 3] = '\0';
    }

    /* Strip trailing whitespace */
    len = strlen(msg);
    while (len > 0 && (msg[len - 1] == ' ' || msg[len - 1] == '\n' ||
                       msg[len - 1] == '\r' || msg[len - 1] == '\t'))
        msg[--len] = '\0';
}

/* ── Print Banner ───────────────────────────────────────────────────── */

static void print_banner(int files, int ins, int del, size_t diff_size, const char *model) {
    printf("\n");
    printf("  %s%scommitaccino%s\n", C_BOLD, C_CYAN, C_RESET);
    printf("  %s───────────────────────────────────%s\n", C_DIM, C_RESET);
    printf("  %s Files changed  %s%d%s\n",   C_DIM, C_RESET, files, C_RESET);
    printf("  %s Insertions     %s+%d%s\n",   C_DIM, C_GREEN, ins, C_RESET);
    printf("  %s Deletions      %s-%d%s\n",   C_DIM, C_RED, del, C_RESET);

    if (diff_size < 1024)
        printf("  %s Diff size      %s%zu B%s\n",    C_DIM, C_RESET, diff_size, C_RESET);
    else
        printf("  %s Diff size      %s%.1f KB%s\n",  C_DIM, C_RESET, diff_size / 1024.0, C_RESET);

    printf("  %s Model          %s%s%s\n",    C_DIM, C_CYAN, model, C_RESET);
    printf("  %s───────────────────────────────────%s\n", C_DIM, C_RESET);
}

/* ── Usage ──────────────────────────────────────────────────────────── */

static void print_usage(void) {
    printf("\n");
    printf("  %s%scommitaccino%s %sv%s%s\n\n", C_BOLD, C_CYAN, C_RESET, C_DIM, VERSION, C_RESET);
    printf("  AI-powered git commits via AWS Bedrock\n\n");
    printf("  %sUsage:%s\n", C_BOLD, C_RESET);
    printf("    commitaccino              Stage all, generate message, commit\n");
    printf("    commitaccino --set-model  Change the default model\n");
    printf("    commitaccino --help       Show this help\n");
    printf("    commitaccino --version    Show version\n\n");
    printf("  %sEnvironment:%s\n", C_BOLD, C_RESET);
    printf("    AWS_BEARER_TOKEN_BEDROCK  Bearer token for Bedrock API\n");
    printf("    AWS_REGION                AWS region (e.g. us-east-1)\n\n");
    printf("  %sConfig:%s ~/.commitaccino (stores preferred model ID)\n\n", C_BOLD, C_RESET);
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
    double t_start = time_ms();

    /* Parse CLI args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--set-model") == 0) {
            char model_id[MAX_MODEL_ID];
            prompt_model_selection(model_id, sizeof(model_id));
            return 0;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
        if (strcmp(argv[i], "--version") == 0 || strcmp(argv[i], "-v") == 0) {
            printf("commitaccino v%s\n", VERSION);
            return 0;
        }
        fprintf(stderr, "  %sUnknown option: %s%s\n", C_RED, argv[i], C_RESET);
        print_usage();
        return 1;
    }

    /* Check env vars */
    const char *token = getenv("AWS_BEARER_TOKEN_BEDROCK");
    if (!token || !*token) {
        fprintf(stderr, "\n  %s✗ AWS_BEARER_TOKEN_BEDROCK is not set%s\n\n", C_RED, C_RESET);
        return 1;
    }

    const char *region = getenv("AWS_REGION");
    if (!region || !*region) {
        fprintf(stderr, "\n  %s✗ AWS_REGION is not set%s\n\n", C_RED, C_RESET);
        return 1;
    }

    /* Load or create config */
    char model_id[MAX_MODEL_ID];
    if (!load_config(model_id, sizeof(model_id))) {
        prompt_model_selection(model_id, sizeof(model_id));
    }

    /* Verify git repo */
    char *git_check = run_cmd("git rev-parse --is-inside-work-tree 2>/dev/null");
    if (!git_check || strncmp(git_check, "true", 4) != 0) {
        fprintf(stderr, "\n  %s✗ Not inside a git repository%s\n\n", C_RED, C_RESET);
        free(git_check);
        return 1;
    }
    free(git_check);

    /* Stage all changes */
    printf("\n  %sStaging changes...%s\n", C_DIM, C_RESET);
    if (run_cmd_status("git add -A") != 0) {
        fprintf(stderr, "  %s✗ Failed to stage changes%s\n\n", C_RED, C_RESET);
        return 1;
    }

    /* Get diff stats */
    char *stat_out = run_cmd("git diff --cached --stat");
    if (!stat_out || !*stat_out) {
        printf("  %sNo changes to commit.%s\n\n", C_DIM, C_RESET);
        free(stat_out);
        return 0;
    }

    int files = 0, insertions = 0, deletions = 0;
    parse_diff_stats(stat_out, &files, &insertions, &deletions);
    free(stat_out);

    /* Get full diff */
    char *diff = run_cmd("git diff --cached");
    if (!diff || !*diff) {
        printf("  %sNo changes to commit.%s\n\n", C_DIM, C_RESET);
        free(diff);
        return 0;
    }

    size_t diff_len = strlen(diff);
    int truncated = 0;
    if (diff_len > MAX_DIFF_SIZE) {
        truncated = 1;
        /* Truncate at last complete line before limit */
        size_t cut = MAX_DIFF_SIZE;
        while (cut > 0 && diff[cut] != '\n') cut--;
        if (cut > 0) {
            diff[cut] = '\0';
            diff_len = cut;
        }
    }

    /* Print stats banner */
    print_banner(files, insertions, deletions, diff_len, model_id);

    if (truncated) {
        printf("  %s⚠ Diff truncated to ~100KB for API request%s\n", C_YELLOW, C_RESET);
    }

    /* Call Bedrock API */
    printf("  %sGenerating commit message...%s", C_DIM, C_RESET);
    fflush(stdout);

    double t_api = time_ms();
    long in_tokens = 0, out_tokens = 0;
    char *message = call_bedrock(region, token, model_id, diff, &in_tokens, &out_tokens);
    double api_ms = time_ms() - t_api;

    free(diff);

    if (!message || !*message) {
        fprintf(stderr, "\n  %s✗ Failed to generate commit message%s\n\n", C_RED, C_RESET);
        free(message);
        return 1;
    }

    printf(" %sdone%s (%.2fs)\n", C_GREEN, C_RESET, api_ms / 1000.0);

    /* Clean up message */
    strip_message(message);

    /* Print the commit message */
    printf("\n  %s%sCommit message:%s\n", C_BOLD, C_GREEN, C_RESET);
    printf("  %s───────────────────────────────────%s\n", C_DIM, C_RESET);

    char *line = message;
    while (line && *line) {
        char *nl = strchr(line, '\n');
        if (nl) {
            *nl = '\0';
            printf("  %s\n", line);
            *nl = '\n';
            line = nl + 1;
        } else {
            printf("  %s\n", line);
            break;
        }
    }

    printf("  %s───────────────────────────────────%s\n", C_DIM, C_RESET);

    /* Commit via git commit -F - (pipe message to stdin, avoids shell escaping) */
    FILE *commit_fp = popen("git commit -F -", "w");
    if (!commit_fp) {
        fprintf(stderr, "\n  %s✗ Failed to run git commit%s\n\n", C_RED, C_RESET);
        free(message);
        return 1;
    }
    fwrite(message, 1, strlen(message), commit_fp);
    int commit_status = pclose(commit_fp);
    free(message);

    if (WEXITSTATUS(commit_status) != 0) {
        fprintf(stderr, "\n  %s✗ git commit failed%s\n\n", C_RED, C_RESET);
        return 1;
    }

    double total_ms = time_ms() - t_start;

    /* Print timing stats */
    printf("\n  %s API time       %s%.2fs%s\n", C_DIM, C_CYAN, api_ms / 1000.0, C_RESET);
    if (in_tokens >= 0)
        printf("  %s Input tokens   %s%ld%s\n", C_DIM, C_RESET, in_tokens, C_RESET);
    if (out_tokens >= 0)
        printf("  %s Output tokens  %s%ld%s\n", C_DIM, C_RESET, out_tokens, C_RESET);
    printf("  %s Total time     %s%.2fs%s\n", C_DIM, C_CYAN, total_ms / 1000.0, C_RESET);
    printf("\n  %s%s✓ Committed successfully!%s\n\n", C_BOLD, C_GREEN, C_RESET);

    return 0;
}
