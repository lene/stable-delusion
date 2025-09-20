# CLI client and web server to easily use nano banana image editing model

## Installation

```bash
poetry install
```

## Usage

### CLI

```bash
$ poetry run python nano_api/generate.py \
    --prompt "please make the women in the provided image look affectionately at each other" \
    --image samples/base.png
```

### Web server

#### Start the server
```bash
$ poetry run python nano_api/main.py
```
#### Make a request to the web API
```bash
$ curl -X POST \
    -F "prompt=please make the women in the provided image look affectionately at each other" \
    -F "images=@samples/base_2.png" \
    http://127.0.0.1:5000/generate
```