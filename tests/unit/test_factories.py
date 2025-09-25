"""Unit tests for factory implementations."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nano_api.factories.repository_factory import RepositoryFactory
from nano_api.factories.service_factory import ServiceFactory
from nano_api.repositories.interfaces import (
    ImageRepository, FileRepository, UploadRepository
)
from nano_api.repositories.local_file_repository import LocalFileRepository
from nano_api.repositories.local_image_repository import LocalImageRepository
from nano_api.repositories.upload_repository import LocalUploadRepository
from nano_api.services.file_service import LocalFileService
from nano_api.services.interfaces import (
    FileService as FileServiceInterface
)


class TestRepositoryFactory:
    """Test cases for RepositoryFactory."""

    def test_create_image_repository(self):
        """Test image repository creation."""
        repo = RepositoryFactory.create_image_repository()

        assert isinstance(repo, ImageRepository)
        assert isinstance(repo, LocalImageRepository)

    def test_create_file_repository(self):
        """Test file repository creation."""
        repo = RepositoryFactory.create_file_repository()

        assert isinstance(repo, FileRepository)
        assert isinstance(repo, LocalFileRepository)

    def test_create_upload_repository(self):
        """Test upload repository creation."""
        repo = RepositoryFactory.create_upload_repository()

        assert isinstance(repo, UploadRepository)
        assert isinstance(repo, LocalUploadRepository)

    def test_create_all_repositories(self):
        """Test creating all repositories at once."""
        image_repo, file_repo, upload_repo = RepositoryFactory.create_all_repositories()

        assert isinstance(image_repo, LocalImageRepository)
        assert isinstance(file_repo, LocalFileRepository)
        assert isinstance(upload_repo, LocalUploadRepository)

    def test_repository_instances_are_independent(self):
        """Test that each factory call creates new instances."""
        repo1 = RepositoryFactory.create_image_repository()
        repo2 = RepositoryFactory.create_image_repository()

        assert repo1 is not repo2
        assert type(repo1) is type(repo2)


class TestServiceFactory:
    """Test cases for ServiceFactory."""

    @patch('nano_api.factories.service_factory.RepositoryFactory')
    def test_create_file_service(self, mock_repo_factory):
        """Test file service creation with dependencies."""
        # Mock return value for create_all_repositories
        mock_image_repo = MagicMock()
        mock_file_repo = MagicMock()
        mock_upload_repo = MagicMock()
        mock_repo_factory.create_all_repositories.return_value = (
            mock_image_repo, mock_file_repo, mock_upload_repo
        )

        service = ServiceFactory.create_file_service()

        assert isinstance(service, FileServiceInterface)
        assert isinstance(service, LocalFileService)
        mock_repo_factory.create_all_repositories.assert_called_once()

    @patch('nano_api.factories.service_factory.GeminiImageGenerationService')
    @patch('nano_api.factories.service_factory.RepositoryFactory')
    def test_create_image_generation_service(self, mock_repo_factory,
                                             mock_gemini_service):
        """Test image generation service creation."""
        project_id = "test-project"
        location = "us-central1"
        output_dir = Path("/tmp/test")

        ServiceFactory.create_image_generation_service(
            project_id=project_id,
            location=location,
            output_dir=output_dir
        )

        mock_repo_factory.create_image_repository.assert_called_once()
        mock_gemini_service.create.assert_called_once_with(
            project_id=project_id,
            location=location,
            output_dir=output_dir,
            image_repository=mock_repo_factory.create_image_repository.return_value
        )

    @patch('nano_api.factories.service_factory.GeminiImageGenerationService')
    @patch('nano_api.factories.service_factory.RepositoryFactory')
    def test_create_image_generation_service_defaults(self, mock_repo_factory,
                                                      mock_gemini_service):
        """Test image generation service creation with defaults."""
        ServiceFactory.create_image_generation_service()

        mock_repo_factory.create_image_repository.assert_called_once()
        mock_gemini_service.create.assert_called_once_with(
            project_id=None,
            location=None,
            output_dir=None,
            image_repository=mock_repo_factory.create_image_repository.return_value
        )

    @patch('nano_api.factories.service_factory.VertexAIUpscalingService')
    def test_create_upscaling_service(self, mock_upscaling_service):
        """Test upscaling service creation."""
        project_id = "test-project"
        location = "us-central1"

        ServiceFactory.create_upscaling_service(
            project_id=project_id,
            location=location
        )

        mock_upscaling_service.create.assert_called_once_with(
            project_id=project_id,
            location=location
        )

    @patch('nano_api.factories.service_factory.VertexAIUpscalingService')
    def test_create_upscaling_service_defaults(self, mock_upscaling_service):
        """Test upscaling service creation with defaults."""
        ServiceFactory.create_upscaling_service()

        mock_upscaling_service.create.assert_called_once_with(
            project_id=None,
            location=None
        )

    @patch('nano_api.factories.service_factory.ServiceFactory.create_file_service')
    @patch('nano_api.factories.service_factory.ServiceFactory.create_image_generation_service')
    @patch('nano_api.factories.service_factory.ServiceFactory.create_upscaling_service')
    def test_create_all_services(self, mock_upscaling, mock_generation, mock_file):
        """Test creating all services at once."""
        project_id = "test-project"
        location = "us-central1"
        output_dir = Path("/tmp/test")

        ServiceFactory.create_all_services(
            project_id=project_id,
            location=location,
            output_dir=output_dir
        )

        mock_file.assert_called_once()
        mock_generation.assert_called_once_with(
            project_id=project_id,
            location=location,
            output_dir=output_dir
        )
        mock_upscaling.assert_called_once_with(
            project_id=project_id,
            location=location
        )

    @patch('nano_api.factories.service_factory.ServiceFactory.create_file_service')
    @patch('nano_api.factories.service_factory.ServiceFactory.create_image_generation_service')
    @patch('nano_api.factories.service_factory.ServiceFactory.create_upscaling_service')
    def test_create_all_services_defaults(self, mock_upscaling, mock_generation, mock_file):
        """Test creating all services with defaults."""
        ServiceFactory.create_all_services()

        mock_file.assert_called_once()
        mock_generation.assert_called_once_with(
            project_id=None,
            location=None,
            output_dir=None
        )
        mock_upscaling.assert_called_once_with(
            project_id=None,
            location=None
        )

    def test_service_instances_are_independent(self):
        """Test that each factory call creates new instances."""
        with patch('nano_api.factories.service_factory.RepositoryFactory') as mock_repo_factory:
            # Mock return value for create_all_repositories
            mock_image_repo = MagicMock()
            mock_file_repo = MagicMock()
            mock_upload_repo = MagicMock()
            mock_repo_factory.create_all_repositories.return_value = (
                mock_image_repo, mock_file_repo, mock_upload_repo
            )

            service1 = ServiceFactory.create_file_service()
            service2 = ServiceFactory.create_file_service()

            assert service1 is not service2
            assert type(service1) is type(service2)


class TestFactoryIntegration:
    """Integration tests for factory pattern."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_factory_creates_working_repositories(self, temp_dir):
        """Test that factory-created repositories work correctly."""
        RepositoryFactory.create_image_repository()
        file_repo = RepositoryFactory.create_file_repository()
        RepositoryFactory.create_upload_repository()

        # Test basic functionality
        assert file_repo.exists(temp_dir) is True
        test_dir = temp_dir / "test_subdir"
        created_dir = file_repo.create_directory(test_dir)
        assert created_dir == test_dir
        assert test_dir.exists()

    def test_factory_creates_working_file_service(self, temp_dir):
        """Test that factory-created file service works correctly."""
        file_service = ServiceFactory.create_file_service()

        # Test basic functionality
        assert hasattr(file_service, 'image_repository')
        assert hasattr(file_service, 'file_repository')

        # These should be working repository instances
        assert file_service.file_repository.exists(temp_dir) is True
