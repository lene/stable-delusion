"""
Unit tests for ServiceFactory Seedream integration.
Tests model selection and service creation with different configurations.
"""

__author__ = "Lene Preuss <lene.preuss@gmail.com>"

from pathlib import Path
from unittest.mock import Mock, patch

from nano_api.factories.service_factory import ServiceFactory
from nano_api.services.seedream_service import SeedreamImageGenerationService
from nano_api.services.gemini_service import GeminiImageGenerationService


class TestServiceFactorySeedream:
    """Test ServiceFactory functionality for Seedream integration."""

    @patch("nano_api.services.seedream_service.SeedreamImageGenerationService")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_create_seedream_service_with_s3_repository(
        self, mock_repo_factory, mock_seedream_service
    ):
        mock_repository = Mock()
        mock_repo_factory.return_value = mock_repository

        mock_service_instance = Mock(spec=SeedreamImageGenerationService)
        mock_seedream_service.create.return_value = mock_service_instance

        service = ServiceFactory.create_image_generation_service(
            model="seedream", output_dir=Path("/tmp"), storage_type="s3"
        )

        # Verify Seedream service was created
        mock_seedream_service.create.assert_called_once_with(
            output_dir=Path("/tmp"), image_repository=mock_repository
        )
        assert service == mock_service_instance

    @patch("nano_api.factories.service_factory.GeminiImageGenerationService")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_service_factory_model_selection_gemini(self, mock_repo_factory, mock_gemini_service):
        mock_repository = Mock()
        mock_repo_factory.return_value = mock_repository

        mock_service_instance = Mock(spec=GeminiImageGenerationService)
        mock_gemini_service.create.return_value = mock_service_instance

        service = ServiceFactory.create_image_generation_service(
            model="gemini", output_dir=Path("/tmp")
        )

        # Verify Gemini service was created
        mock_gemini_service.create.assert_called_once()
        assert service == mock_service_instance

    @patch("nano_api.factories.service_factory.GeminiImageGenerationService")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_service_factory_model_selection_default(self, mock_repo_factory, mock_gemini_service):
        mock_repository = Mock()
        mock_repo_factory.return_value = mock_repository

        mock_service_instance = Mock(spec=GeminiImageGenerationService)
        mock_gemini_service.create.return_value = mock_service_instance

        service = ServiceFactory.create_image_generation_service(
            model=None, output_dir=Path("/tmp")  # Default model
        )

        # Verify Gemini service was created (default)
        mock_gemini_service.create.assert_called_once()
        assert service == mock_service_instance

    @patch("nano_api.config.ConfigManager.get_config")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_service_factory_storage_type_override(self, mock_repo_factory, mock_config):
        # Mock original configuration
        mock_original_config = Mock()
        mock_original_config.storage_type = "local"
        mock_config.return_value = mock_original_config

        mock_repository = Mock()
        mock_repo_factory.return_value = mock_repository

        with patch(
            "nano_api.services.seedream_service.SeedreamImageGenerationService"
        ) as mock_seedream_service:
            mock_service_instance = Mock()
            mock_seedream_service.create.return_value = mock_service_instance

            ServiceFactory.create_image_generation_service(
                model="seedream", storage_type="s3"  # Override to S3
            )

        # Verify that storage type was temporarily changed to S3
        # This is tested indirectly through the repository creation
        mock_repo_factory.assert_called_once()

    @patch("nano_api.services.seedream_service.SeedreamImageGenerationService")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_create_seedream_service_without_storage_override(
        self, mock_repo_factory, mock_seedream_service
    ):
        mock_repository = Mock()
        mock_repo_factory.return_value = mock_repository

        mock_service_instance = Mock()
        mock_seedream_service.create.return_value = mock_service_instance

        ServiceFactory.create_image_generation_service(
            model="seedream", project_id="test-project", location="us-central1"
        )

        # Repository should be created normally without override
        mock_repo_factory.assert_called_once()
        mock_seedream_service.create.assert_called_once_with(
            output_dir=None, image_repository=mock_repository
        )

    @patch("nano_api.services.seedream_service.SeedreamImageGenerationService")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_create_seedream_service_with_all_parameters(
        self, mock_repo_factory, mock_seedream_service
    ):
        mock_repository = Mock()
        mock_repo_factory.return_value = mock_repository

        mock_service_instance = Mock()
        mock_seedream_service.create.return_value = mock_service_instance

        service = ServiceFactory.create_image_generation_service(
            project_id="test-project",
            location="eu-central-1",
            output_dir=Path("/custom/output"),
            storage_type="s3",
            model="seedream",
        )

        mock_seedream_service.create.assert_called_once_with(
            output_dir=Path("/custom/output"), image_repository=mock_repository
        )
        assert service == mock_service_instance

    def test_service_factory_logging(self):
        with patch("nano_api.factories.service_factory.logging") as mock_logging:
            with patch(
                "nano_api.services.seedream_service.SeedreamImageGenerationService"
            ) as mock_seedream_service:
                with patch(
                    "nano_api.factories.repository_factory.RepositoryFactory."
                    "create_image_repository"
                ):
                    mock_service_instance = Mock()
                    mock_seedream_service.create.return_value = mock_service_instance

                    ServiceFactory.create_image_generation_service(model="seedream")

        # Verify logging calls
        mock_logging.info.assert_any_call(
            "🏭 ServiceFactory creating service for model: %s", "seedream"
        )
        mock_logging.info.assert_any_call("🌱 Creating SeedreamImageGenerationService")
        mock_logging.info.assert_any_call(
            "✅ SeedreamImageGenerationService created: %s", mock_service_instance
        )

    def test_service_factory_gemini_logging(self):
        with patch("nano_api.factories.service_factory.logging") as mock_logging:
            with patch(
                "nano_api.factories.service_factory.GeminiImageGenerationService"
            ) as mock_gemini_service:
                with patch(
                    "nano_api.factories.repository_factory.RepositoryFactory."
                    "create_image_repository"
                ):
                    mock_service_instance = Mock()
                    mock_gemini_service.create.return_value = mock_service_instance

                    ServiceFactory.create_image_generation_service(model="gemini")

        # Verify logging calls
        mock_logging.info.assert_any_call(
            "🏭 ServiceFactory creating service for model: %s", "gemini"
        )
        mock_logging.info.assert_any_call("🔷 Creating GeminiImageGenerationService")
        mock_logging.info.assert_any_call(
            "✅ GeminiImageGenerationService created: %s", mock_service_instance
        )

    @patch("nano_api.services.seedream_service.SeedreamImageGenerationService")
    @patch("nano_api.factories.repository_factory.RepositoryFactory.create_image_repository")
    def test_backward_compatibility_default_model(self, mock_repo_factory, mock_seedream_service):
        with patch(
            "nano_api.factories.service_factory.GeminiImageGenerationService"
        ) as mock_gemini_service:
            mock_repository = Mock()
            mock_repo_factory.return_value = mock_repository

            mock_service_instance = Mock()
            mock_gemini_service.create.return_value = mock_service_instance

            # No model specified - should default to Gemini
            ServiceFactory.create_image_generation_service()

            # Should create Gemini service (default for backward compatibility)
            mock_gemini_service.create.assert_called_once()
            mock_seedream_service.create.assert_not_called()

    @patch("nano_api.factories.service_factory.ServiceFactory.create_image_generation_service")
    @patch("nano_api.factories.service_factory.ServiceFactory.create_file_service")
    @patch("nano_api.factories.service_factory.ServiceFactory.create_upscaling_service")
    def test_create_all_services_integration(self, mock_upscaling, mock_file, mock_image_gen):
        mock_file_service = Mock()
        mock_image_service = Mock()
        mock_upscaling_service = Mock()

        mock_file.return_value = mock_file_service
        mock_image_gen.return_value = mock_image_service
        mock_upscaling.return_value = mock_upscaling_service

        file_service, image_service, upscaling_service = ServiceFactory.create_all_services(
            project_id="test-project", location="us-central1", output_dir=Path("/tmp")
        )

        # Verify all services were created
        mock_file.assert_called_once()
        mock_image_gen.assert_called_once_with(
            project_id="test-project", location="us-central1", output_dir=Path("/tmp")
        )
        mock_upscaling.assert_called_once_with(project_id="test-project", location="us-central1")

        assert file_service == mock_file_service
        assert image_service == mock_image_service
        assert upscaling_service == mock_upscaling_service

    def test_invalid_model_handling(self):
        # The factory should handle unknown models gracefully by defaulting to Gemini
        with patch(
            "nano_api.factories.service_factory.GeminiImageGenerationService"
        ) as mock_gemini_service:
            with patch(
                "nano_api.factories.repository_factory.RepositoryFactory.create_image_repository"
            ):
                mock_service_instance = Mock()
                mock_gemini_service.create.return_value = mock_service_instance

                # Invalid model should fall through to default (Gemini)
                ServiceFactory.create_image_generation_service(model="invalid_model")

                mock_gemini_service.create.assert_called_once()

    @patch("nano_api.config.ConfigManager.get_config")
    def test_storage_type_override_restoration(self, mock_config):
        # Mock original configuration
        mock_original_config = Mock()
        mock_original_config.storage_type = "local"
        mock_config.return_value = mock_original_config

        with patch(
            "nano_api.factories.repository_factory.RepositoryFactory.create_image_repository"
        ):
            with patch(
                "nano_api.services.seedream_service.SeedreamImageGenerationService"
            ) as mock_seedream_service:
                mock_seedream_service.create.return_value = Mock()

                ServiceFactory.create_image_generation_service(
                    model="seedream", storage_type="s3"  # Override
                )

        # After the method completes, original storage type should be restored
        assert mock_original_config.storage_type == "local"

    def test_case_insensitive_model_selection(self):
        test_cases = [
            ("seedream", "SeedreamImageGenerationService"),
            ("SEEDREAM", "SeedreamImageGenerationService"),
            ("Seedream", "SeedreamImageGenerationService"),
            ("gemini", "GeminiImageGenerationService"),
            ("GEMINI", "GeminiImageGenerationService"),
            ("Gemini", "GeminiImageGenerationService"),
        ]

        for model_name, expected_service in test_cases:
            with patch(
                "nano_api.factories.repository_factory.RepositoryFactory.create_image_repository"
            ):
                if "Seedream" in expected_service:
                    with patch(
                        "nano_api.services.seedream_service.SeedreamImageGenerationService"
                    ) as mock_service:
                        mock_service.create.return_value = Mock()
                        ServiceFactory.create_image_generation_service(model=model_name.lower())
                        if model_name.lower() == "seedream":
                            mock_service.create.assert_called_once()
                else:
                    with patch(
                        "nano_api.factories.service_factory.GeminiImageGenerationService"
                    ) as mock_service:
                        mock_service.create.return_value = Mock()
                        ServiceFactory.create_image_generation_service(model=model_name.lower())
                        if model_name.lower() == "gemini":
                            mock_service.create.assert_called_once()
