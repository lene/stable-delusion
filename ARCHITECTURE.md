# NanoAPIClient System Architecture

This diagram shows the complete system architecture including all components, data flows, and external services.

```mermaid
graph TB
    %% User Interfaces
    CLI[CLI Interface<br/>nano_api/generate.py]
    WebUI[Web Frontend<br/>Planned - Issue #1]

    %% Core Application Layer
    Flask[Flask API Server<br/>nano_api/main.py]
    GeminiClient[GeminiClient Class<br/>nano_api/generate.py]
    Utils[Utility Functions<br/>nano_api/utils.py]
    Upscale[Upscaling Module<br/>nano_api/upscale.py]
    Config[Configuration<br/>nano_api/conf.py]

    %% External APIs
    GeminiAPI[Google Gemini API<br/>gemini-2.5-flash-image-preview]
    VertexAI[Google Vertex AI<br/>Imagen Upscaling]

    %% File System
    LocalFS[Local File System<br/>uploads/, outputs/]

    %% API Documentation
    OpenAPI[OpenAPI Specification<br/>openapi.json]

    %% User Interactions
    CLI -->|Image Generation| GeminiClient
    WebUI -.->|HTTP Requests| Flask

    %% Flask API Endpoints
    Flask -->|POST /generate| GeminiClient
    Flask -->|GET /health| Utils
    Flask -->|GET /| Utils
    Flask -->|GET /openapi.json| OpenAPI

    %% Core Processing Flow
    GeminiClient -->|Upload Images| GeminiAPI
    GeminiClient -->|Generate Content| GeminiAPI
    GeminiClient -->|Optional Upscaling| Upscale

    %% Upscaling Flow
    Upscale -->|Upscale Request| VertexAI
    VertexAI -->|Upscaled Image| Upscale

    %% Utility Functions
    GeminiClient -->|Logging & Timestamps| Utils
    Flask -->|Error Handling| Utils

    %% Configuration
    GeminiClient -->|Project Settings| Config
    Upscale -->|Project Settings| Config

    %% File Operations
    GeminiClient -->|Save Results| LocalFS
    Flask -->|Store Uploads| LocalFS
    CLI -->|Read Reference Images| LocalFS

    %% External Service Flows
    GeminiAPI -->|Generated Images| GeminiClient

    %% Styling
    classDef userInterface fill:#e1f5fe
    classDef coreModule fill:#f3e5f5
    classDef externalAPI fill:#fff3e0
    classDef storage fill:#e8f5e8
    classDef documentation fill:#fce4ec

    class CLI,WebUI userInterface
    class Flask,GeminiClient,Utils,Upscale,Config coreModule
    class GeminiAPI,VertexAI externalAPI
    class LocalFS storage
    class OpenAPI documentation
```

## Component Details

### User Interfaces
- **CLI Interface** (`nano_api/generate.py`): Command-line tool with full parameter support
- **Web Frontend** (Planned): Modern web UI for image generation (GitLab Issue #1)

### Core Application Layer
- **Flask API Server** (`nano_api/main.py`): REST API with endpoints for generation, health, and documentation
- **GeminiClient Class** (`nano_api/generate.py`): Core logic for AI image generation
- **Utility Functions** (`nano_api/utils.py`): Shared utilities for timestamps, logging, error handling
- **Upscaling Module** (`nano_api/upscale.py`): Image upscaling using Vertex AI
- **Configuration** (`nano_api/conf.py`): Default project settings and constants

### External Services
- **Google Gemini API**: AI image generation using gemini-2.5-flash-image-preview model
- **Google Vertex AI**: Image upscaling using Imagen model (2x, 4x scaling)

### Data Flow
1. **Input**: Users provide prompts and reference images via CLI or API
2. **Processing**: Images uploaded to Gemini API, content generated
3. **Optional Upscaling**: Generated images processed through Vertex AI
4. **Output**: Final images saved to local filesystem with timestamps

### API Endpoints
- `POST /generate`: Main image generation with all CLI parameters
- `GET /health`: Service health check
- `GET /`: API information and endpoint documentation
- `GET /openapi.json`: Complete OpenAPI 3.0.3 specification

### Key Features
- **Multi-image Support**: Multiple reference images per generation
- **Parameter Flexibility**: Full CLI parameter support in API
- **Error Handling**: Comprehensive logging and user feedback
- **Security**: Environment-based configuration, secure file handling
- **Quality Assurance**: Comprehensive testing, linting, type checking
- **Documentation**: OpenAPI specification, API demo examples

### Technology Stack
- **Backend**: Python 3.10, Flask, Poetry dependency management
- **AI Services**: Google Gemini API, Google Vertex AI
- **Testing**: pytest (61 tests), comprehensive coverage
- **Quality Tools**: pylint, flake8, mypy, bandit
- **CI/CD**: GitLab CI/CD with automated quality gates
- **Documentation**: OpenAPI 3.0.3, Markdown documentation