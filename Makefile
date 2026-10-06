
all: run

install:
	@echo "Installing dependencies..."
	uv sync

run:
	@echo "Running..."
	uv run python -m src

debug:
	@echo "Debugging..."

clean:
	@echo "Cleaning up..."

lint:
	@echo "Linting..."
	
lint-strict:
	@echo "Linting strictly..."