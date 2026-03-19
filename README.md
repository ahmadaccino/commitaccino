# commitaccino

Fast, AI-powered git commits from your terminal. Reads your staged diff, generates a commit message via AWS Bedrock, and commits — all in one command.

Written in C with zero runtime dependencies beyond libcurl. Typically completes in under 2 seconds.

## Install

### Homebrew

```bash
brew install ahmadaccino/tap/commitaccino
```

### From source

Requires `libcurl` (included on macOS and most Linux distros).

```bash
git clone https://github.com/ahmadaccino/commitaccino.git
cd commitaccino
make
sudo make install
```

This installs to `/usr/local/bin`. Override with `PREFIX`:

```bash
make install PREFIX=~/.local/bin
```

## Setup

commitaccino uses the [AWS Bedrock Converse API](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html) and requires two environment variables:

```bash
export AWS_BEARER_TOKEN_BEDROCK="your-token-here"
export AWS_REGION="us-east-1"
```

Add these to your shell profile (`~/.zshrc`, `~/.bashrc`, etc.).

## Usage

```bash
# Stage all changes, generate message, and commit
commitaccino

# Choose your preferred model
commitaccino --set-model
```

On first run, you'll be prompted to pick a default model. Your choice is saved to `~/.commitaccino`.

### Supported models

| # | Model |
|---|-------|
| 1 | Claude Sonnet 4 |
| 2 | Claude Haiku 4 |
| 3 | Amazon Nova Micro |
| 4 | Amazon Nova Lite |
| 5 | Llama 4 Scout 17B |

### Example output

```
  commitaccino
  ───────────────────────────────────
   Files changed  3
   Insertions     +42
   Deletions      -7
   Diff size      2.1 KB
   Model          us.anthropic.claude-sonnet-4-20250514-v1:0
  ───────────────────────────────────
  Generating commit message... done (0.84s)

  Commit message:
  ───────────────────────────────────
  feat: add retry logic for transient API failures
  ───────────────────────────────────

   API time       0.84s
   Input tokens   1523
   Output tokens  18
   Total time     1.12s

  ✓ Committed successfully!
```

### Options

```
commitaccino              Stage all, generate message, commit
commitaccino --set-model  Change the default model
commitaccino --help       Show help
commitaccino --version    Show version
```

## How it works

1. Runs `git add -A` to stage all changes
2. Reads the staged diff (`git diff --cached`)
3. Sends the diff to AWS Bedrock's Converse API
4. Cleans up the response and commits via `git commit`

Large diffs (>100KB) are automatically truncated to stay within API limits.

## License

[MIT](LICENSE)
