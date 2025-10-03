# Fish Shell Environment Loading

## Quick Start

Load environment variables from `.env` file in fish shell:

```fish
# From project root
source scripts/load_env.fish

# Or with a custom .env file
source scripts/load_env.fish path/to/custom.env
```

## Alternative Methods

### Method 1: Add to your fish config (Recommended for this project)

Add to `~/.config/fish/config.fish`:

```fish
# Auto-load .env when entering the project directory
function __load_stable_delusion_env --on-variable PWD
    if test -f "$PWD/.env"
        and string match -q "*/NanoAPIClient" "$PWD"
        source "$PWD/scripts/load_env.fish"
    end
end
```

This will automatically load `.env` whenever you `cd` into the project directory.

### Method 2: Create an alias

Add to `~/.config/fish/config.fish`:

```fish
alias load-env='source scripts/load_env.fish'
```

Then just run:
```fish
load-env
```

### Method 3: One-liner for quick loading

```fish
# Export all KEY=VALUE pairs from .env
while read -l line
    set -l parts (string split -m 1 '=' -- $line)
    test (count $parts) -eq 2; and set -gx $parts[1] $parts[2]
end < .env
```

### Method 4: Use fisher plugin (if you have fisher installed)

```fish
fisher install edc/bass
bass source .env
```

## What the script does

The `load_env.fish` script:
- ✅ Reads `.env` file line by line
- ✅ Skips empty lines and comments (`#`)
- ✅ Parses `KEY=VALUE` format
- ✅ Removes quotes from values
- ✅ Exports variables globally (`set -gx`)
- ✅ Shows which variables were loaded

## Example .env format

```bash
# Database configuration
DATABASE_URL=postgresql://localhost/mydb
DATABASE_PORT=5432

# API Keys
GEMINI_API_KEY="your-api-key-here"
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# S3 Configuration
AWS_S3_BUCKET=my-bucket-name
AWS_S3_REGION=us-east-1
```

## Verify variables are loaded

```fish
echo $GEMINI_API_KEY
echo $AWS_S3_BUCKET
```

## Troubleshooting

**Variables not persisting after script runs?**
- Make sure you're using `source` not just running the script
- Use `source scripts/load_env.fish` not `./scripts/load_env.fish`

**Quote handling issues?**
- The script removes both single and double quotes
- If you need quotes in the value, escape them: `KEY="value with \"quotes\""`

**Variable not exported to child processes?**
- The script uses `set -gx` which exports globally
- Verify with: `fish -c 'echo $YOUR_VAR'`
