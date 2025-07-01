# Use .PHONY to ensure these targets run even if files with the same name exist.
.PHONY: all clean build test publish-test publish

# Define variables for commands and repositories
PYTHON := python
BUILD_COMMAND := $(PYTHON) -m build
TWINE := $(PYTHON) -m twine

# Default target when just 'make' is run
all: build

# Clean up build artifacts and caches
clean:
    @echo "Cleaning __pycache__ directories..."
    find . -type d -name "__pycache__" -exec rm -rf {} +
    @echo "Cleaning build and distribution directories..."
    rm -rf build dist *.egg-info

# Build the sdist and wheel. Depends on 'clean' to ensure a fresh build.
build: clean
    @echo "Building the package..."
    $(BUILD_COMMAND)
    @echo "Build process completed. Artifacts are in dist/"

# A placeholder for your test suite (e.g., using pytest)
test:
    @echo "Running tests..."
    $(PYTHON) -m pytest

# --- Publishing Targets ---

# Target to upload to TestPyPI for staging
publish-test: build
    @echo "Uploading package to TestPyPI..."
    $(TWINE) upload --repository testpypi dist/*

# Target to upload to the official PyPI
# This is the "production" release command.
publish: build
	@read -p "You are about to upload to the official PyPI. This is IRREVERSIBLE. Are you sure? (y/N) " confirm && \
	if [ "$$confirm" = "y" ]; then \
		echo "Uploading package to PyPI..."; \
		$(TWINE) upload dist/*; \
	else \
		echo "Upload cancelled."; \
	fi