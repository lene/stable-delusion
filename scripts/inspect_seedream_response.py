#!/usr/bin/env python3
"""
Script to inspect the full structure of Seedream API responses.
Makes a real API call and dumps all response attributes.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_delusion.seedream import SeedreamClient


def inspect_response_structure():
    """Make a Seedream API call and inspect the full response."""

    # Initialize client
    print("Initializing Seedream client...")
    client = SeedreamClient.create_with_env_key()

    # S3 image URL with correct region (eu-central-1)
    image_url = "https://nano-api-generated-images-lene.s3.eu-central-1.amazonaws.com/input/seedream_input_102570881_hqcelebcorner__6__2025-09-28-19:17:29.jpg"
    prompt = "make this image more beautiful"

    print(f"\nMaking API request...")
    print(f"Prompt: {prompt}")
    print(f"Image URL: {image_url}")

    # Make direct API call to get raw response
    try:
        response = client.client.images.generate(
            model=client.model,
            prompt=prompt,
            image=[image_url],
            size="2K",
            sequential_image_generation="disabled",
            response_format="url",
            watermark=False,
        )

        print("\n" + "="*80)
        print("RESPONSE TYPE:", type(response))
        print("="*80)

        print("\n" + "="*80)
        print("RESPONSE ATTRIBUTES:")
        print("="*80)
        for attr in dir(response):
            if not attr.startswith('_'):
                print(f"  {attr}")

        print("\n" + "="*80)
        print("RESPONSE VALUES:")
        print("="*80)
        for attr in dir(response):
            if not attr.startswith('_') and not callable(getattr(response, attr)):
                try:
                    value = getattr(response, attr)
                    print(f"\n{attr}:")
                    print(f"  Type: {type(value)}")
                    print(f"  Value: {value}")
                except Exception as e:
                    print(f"\n{attr}: [Error accessing: {e}]")

        # Try to convert to dict if method exists
        print("\n" + "="*80)
        print("ATTEMPTING DICT CONVERSION:")
        print("="*80)

        if hasattr(response, 'model_dump'):
            print("\nUsing model_dump():")
            response_dict = response.model_dump()
            print(json.dumps(response_dict, indent=2, default=str))
        elif hasattr(response, 'dict'):
            print("\nUsing dict():")
            response_dict = response.dict()
            print(json.dumps(response_dict, indent=2, default=str))
        elif hasattr(response, '__dict__'):
            print("\nUsing __dict__:")
            response_dict = response.__dict__
            print(json.dumps(response_dict, indent=2, default=str))
        else:
            print("\nNo dict conversion method available")

        # Inspect the data attribute specifically
        if hasattr(response, 'data') and response.data:
            print("\n" + "="*80)
            print("DATA ITEMS INSPECTION:")
            print("="*80)
            for i, item in enumerate(response.data):
                print(f"\nItem {i}:")
                print(f"  Type: {type(item)}")
                for attr in dir(item):
                    if not attr.startswith('_') and not callable(getattr(item, attr)):
                        try:
                            value = getattr(item, attr)
                            print(f"  {attr}: {value} (type: {type(value)})")
                        except Exception as e:
                            print(f"  {attr}: [Error: {e}]")

        return response

    except Exception as e:
        print(f"\n❌ Error making API request: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    inspect_response_structure()
