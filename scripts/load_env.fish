#!/usr/bin/env fish
# Load environment variables from .env file for fish shell

function load_env --description 'Load environment variables from .env file'
    set -l env_file ".env"

    # Allow custom env file path
    if test (count $argv) -gt 0
        set env_file $argv[1]
    end

    if not test -f $env_file
        echo "❌ Error: $env_file not found"
        return 1
    end

    echo "📄 Loading environment variables from $env_file"
    echo ""

    # Read the .env file line by line
    while read -l line
        # Skip empty lines
        if test -z "$line"
            continue
        end

        # Skip comments
        if string match -q -r '^\s*#' -- $line
            continue
        end

        # Parse KEY=VALUE format
        if string match -q -r '^[A-Za-z_][A-Za-z0-9_]*=' -- $line
            # Split on first = only
            set -l key (string split -m 1 '=' -- $line)[1]
            set -l value (string split -m 1 '=' -- $line)[2]

            # Remove quotes if present
            set value (string trim -c '\'"' -- $value)

            # Export the variable
            set -gx $key $value
            echo "✅ $key=$value"
        end
    end < $env_file

    echo ""
    echo "✨ Environment variables loaded successfully!"
end

# Run the function if script is executed directly
load_env $argv
