# Functorial Narrative Analysis - Research Makefile
# ================================================
# This Makefile provides reproducible commands for the entire research pipeline.
# Run `make help` for a list of available commands.

.PHONY: help setup install test lint clean \
        corpus-gutenberg corpus-ao3 corpus-syosetu corpus-all \
        corpus-ancient corpus-streaming \
        preprocess extract-trajectories cluster analyze \
        extract-entropy analyze-gutenberg-parenthesis analyze-dopamine \
        analyze-gilgamesh analyze-entropy \
        report visualize all

# Configuration
PYTHON := python3
PIP := pip3
VENV := .venv
DATA_DIR := data
SRC_DIR := src
NOTEBOOKS_DIR := notebooks
RESULTS_DIR := data/results

# Corpus sizes (adjust for development vs production)
SAMPLE_SIZE ?= 1000
N_CLUSTERS ?= 6

# Colors for terminal output
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

#------------------------------------------------------------------------------
# HELP
#------------------------------------------------------------------------------
help:
	@echo "$(GREEN)Functorial Narrative Analysis - Research Commands$(NC)"
	@echo "=================================================="
	@echo ""
	@echo "$(YELLOW)Setup & Environment:$(NC)"
	@echo "  make setup          - Create virtual environment and install dependencies"
	@echo "  make install        - Install Python dependencies only"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Run code linting"
	@echo ""
	@echo "$(YELLOW)Corpus Collection (Phase 1-3):$(NC)"
	@echo "  make corpus-gutenberg      - Download and prepare Project Gutenberg corpus"
	@echo "  make corpus-ao3            - Collect AO3 fan fiction corpus"
	@echo "  make corpus-syosetu        - Collect Japanese web novels from Syosetu"
	@echo "  make corpus-opensubtitles  - Download OpenSubtitles film corpus"
	@echo "  make corpus-ancient        - Collect ancient epics (Gilgamesh, Homer, etc.)"
	@echo "  make corpus-streaming      - Collect streaming-era TV content"
	@echo "  make corpus-all            - Collect all corpora"
	@echo ""
	@echo "$(YELLOW)Preprocessing (Phase 2):$(NC)"
	@echo "  make preprocess            - Run full preprocessing pipeline"
	@echo "  make preprocess-segment    - Sentence segmentation only"
	@echo "  make preprocess-window     - Create sliding windows"
	@echo "  make preprocess-normalize  - Normalize to narrative time"
	@echo ""
	@echo "$(YELLOW)Functor Extraction (Phase 4):$(NC)"
	@echo "  make extract-trajectories  - Extract all functor trajectories"
	@echo "  make extract-sentiment     - Run F_sentiment functor"
	@echo "  make extract-arousal       - Run F_arousal functor"
	@echo "  make extract-epistemic     - Run F_epistemic functor"
	@echo "  make extract-thematic      - Run F_thematic functor"
	@echo "  make extract-entropy       - Run F_entropy (Shannon) functor"
	@echo ""
	@echo "$(YELLOW)Structural Detection (Phase 4):$(NC)"
	@echo "  make detect-harmon         - Detect Harmon Circle structure"
	@echo "  make detect-kishotenketsu  - Detect kishōtenketsu structure"
	@echo "  make detect-all            - Run all structural detectors"
	@echo ""
	@echo "$(YELLOW)Clustering & Analysis (Phase 4-5):$(NC)"
	@echo "  make cluster               - Cluster trajectories by shape"
	@echo "  make analyze-transfer      - Cross-cultural transfer analysis"
	@echo "  make analyze-temporal      - Temporal drift analysis"
	@echo "  make analyze-editorial     - Editorial gatekeeping analysis"
	@echo "  make analyze-all           - Run full analysis suite"
	@echo ""
	@echo "$(YELLOW)Extended Analysis (Q8-Q11):$(NC)"
	@echo "  make analyze-gutenberg-parenthesis  - Q8: Test Gutenberg Parenthesis hypothesis"
	@echo "  make analyze-dopamine               - Q9: Dopamine optimization analysis"
	@echo "  make analyze-gilgamesh              - Q10: Ancient epic structure analysis"
	@echo "  make analyze-entropy                - Q11: Shannon entropy patterns"
	@echo "  make analyze-extended               - Run all extended analyses"
	@echo ""
	@echo "$(YELLOW)Reporting (Phase 5):$(NC)"
	@echo "  make visualize             - Generate all visualizations"
	@echo "  make report                - Generate research report"
	@echo "  make export-results        - Export results to CSV/JSON"
	@echo ""
	@echo "$(YELLOW)Full Pipeline:$(NC)"
	@echo "  make all                   - Run entire research pipeline"
	@echo "  make replication           - Full replication from scratch"
	@echo ""
	@echo "$(YELLOW)Utilities:$(NC)"
	@echo "  make clean                 - Remove generated files"
	@echo "  make clean-data            - Remove downloaded data"
	@echo "  make notebook              - Start Jupyter notebook server"
	@echo "  make validate              - Validate data integrity"
	@echo ""

#------------------------------------------------------------------------------
# SETUP & ENVIRONMENT
#------------------------------------------------------------------------------
setup: $(VENV)/bin/activate install
	@echo "$(GREEN)✓ Environment setup complete$(NC)"

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Virtual environment created$(NC)"

install: requirements.txt
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install requirements-dev.txt
	$(VENV)/bin/pip install -r requirements-dev.txt
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

requirements.txt:
	@echo "Creating requirements.txt..."
	@echo "# Functorial Narrative Analysis Dependencies" > requirements.txt
	@echo "# Core" >> requirements.txt
	@echo "numpy>=1.24.0" >> requirements.txt
	@echo "pandas>=2.0.0" >> requirements.txt
	@echo "scipy>=1.10.0" >> requirements.txt
	@echo "scikit-learn>=1.2.0" >> requirements.txt
	@echo "" >> requirements.txt
	@echo "# NLP & Transformers" >> requirements.txt
	@echo "transformers>=4.30.0" >> requirements.txt
	@echo "torch>=2.0.0" >> requirements.txt
	@echo "sentence-transformers>=2.2.0" >> requirements.txt
	@echo "vaderSentiment>=3.3.2" >> requirements.txt
	@echo "nltk>=3.8.0" >> requirements.txt
	@echo "spacy>=3.5.0" >> requirements.txt
	@echo "" >> requirements.txt
	@echo "# Time Series & Clustering" >> requirements.txt
	@echo "fastdtw>=0.3.4" >> requirements.txt
	@echo "tslearn>=0.5.3" >> requirements.txt
	@echo "" >> requirements.txt
	@echo "# Visualization" >> requirements.txt
	@echo "matplotlib>=3.7.0" >> requirements.txt
	@echo "seaborn>=0.12.0" >> requirements.txt
	@echo "plotly>=5.14.0" >> requirements.txt
	@echo "umap-learn>=0.5.3" >> requirements.txt
	@echo "" >> requirements.txt
	@echo "# Data Collection" >> requirements.txt
	@echo "requests>=2.28.0" >> requirements.txt
	@echo "beautifulsoup4>=4.12.0" >> requirements.txt
	@echo "aiohttp>=3.8.0" >> requirements.txt
	@echo "tqdm>=4.65.0" >> requirements.txt
	@echo "" >> requirements.txt
	@echo "# Utilities" >> requirements.txt
	@echo "python-dotenv>=1.0.0" >> requirements.txt
	@echo "pyyaml>=6.0" >> requirements.txt
	@echo "click>=8.1.0" >> requirements.txt
	@echo "rich>=13.0.0" >> requirements.txt
	@echo "" >> requirements.txt
	@echo "# Jupyter" >> requirements.txt
	@echo "jupyter>=1.0.0" >> requirements.txt
	@echo "jupyterlab>=4.0.0" >> requirements.txt
	@echo "ipywidgets>=8.0.0" >> requirements.txt

requirements-dev.txt:
	@echo "# Development Dependencies" > requirements-dev.txt
	@echo "pytest>=7.3.0" >> requirements-dev.txt
	@echo "pytest-cov>=4.0.0" >> requirements-dev.txt
	@echo "black>=23.3.0" >> requirements-dev.txt
	@echo "isort>=5.12.0" >> requirements-dev.txt
	@echo "flake8>=6.0.0" >> requirements-dev.txt
	@echo "mypy>=1.3.0" >> requirements-dev.txt
	@echo "pre-commit>=3.3.0" >> requirements-dev.txt

test:
	$(VENV)/bin/pytest tests/ -v --cov=$(SRC_DIR) --cov-report=term-missing

lint:
	$(VENV)/bin/black $(SRC_DIR) tests/
	$(VENV)/bin/isort $(SRC_DIR) tests/
	$(VENV)/bin/flake8 $(SRC_DIR) tests/

#------------------------------------------------------------------------------
# CORPUS COLLECTION
#------------------------------------------------------------------------------
corpus-gutenberg:
	@echo "$(YELLOW)Downloading Project Gutenberg corpus...$(NC)"
	$(VENV)/bin/python -m src.corpus.gutenberg \
		--output $(DATA_DIR)/raw/gutenberg \
		--sample-size $(SAMPLE_SIZE)
	@echo "$(GREEN)✓ Gutenberg corpus downloaded$(NC)"

corpus-ao3:
	@echo "$(YELLOW)Collecting AO3 corpus...$(NC)"
	$(VENV)/bin/python -m src.corpus.ao3 \
		--output $(DATA_DIR)/raw/ao3 \
		--sample-size $(SAMPLE_SIZE) \
		--fandoms "Harry Potter,Naruto,MCU,BTS"
	@echo "$(GREEN)✓ AO3 corpus collected$(NC)"

corpus-syosetu:
	@echo "$(YELLOW)Collecting Syosetu (Japanese web novels) corpus...$(NC)"
	$(VENV)/bin/python -m src.corpus.syosetu \
		--output $(DATA_DIR)/raw/syosetu \
		--sample-size $(SAMPLE_SIZE)
	@echo "$(GREEN)✓ Syosetu corpus collected$(NC)"

corpus-opensubtitles:
	@echo "$(YELLOW)Downloading OpenSubtitles corpus...$(NC)"
	$(VENV)/bin/python -m src.corpus.opensubtitles \
		--output $(DATA_DIR)/raw/opensubtitles \
		--languages "en,ja,ko,zh" \
		--sample-size $(SAMPLE_SIZE)
	@echo "$(GREEN)✓ OpenSubtitles corpus downloaded$(NC)"

corpus-ancient:
	@echo "$(YELLOW)Collecting ancient epic corpus...$(NC)"
	$(VENV)/bin/python -m src.corpus.ancient \
		--output $(DATA_DIR)/raw/ancient \
		--texts "gilgamesh,iliad,odyssey,mahabharata,beowulf"
	@echo "$(GREEN)✓ Ancient corpus collected$(NC)"

corpus-streaming:
	@echo "$(YELLOW)Collecting streaming-era TV corpus...$(NC)"
	$(VENV)/bin/python -m src.corpus.streaming \
		--output $(DATA_DIR)/raw/streaming \
		--eras "pre-streaming,early-streaming,algorithmic"
	@echo "$(GREEN)✓ Streaming corpus collected$(NC)"

corpus-all: corpus-gutenberg corpus-ao3 corpus-syosetu corpus-opensubtitles corpus-ancient corpus-streaming
	@echo "$(GREEN)✓ All corpora collected$(NC)"

#------------------------------------------------------------------------------
# PREPROCESSING
#------------------------------------------------------------------------------
preprocess: preprocess-segment preprocess-window preprocess-normalize
	@echo "$(GREEN)✓ Preprocessing complete$(NC)"

preprocess-segment:
	@echo "$(YELLOW)Running sentence segmentation...$(NC)"
	$(VENV)/bin/python -m src.corpus.preprocess segment \
		--input $(DATA_DIR)/raw \
		--output $(DATA_DIR)/processed/segmented

preprocess-window:
	@echo "$(YELLOW)Creating sliding windows...$(NC)"
	$(VENV)/bin/python -m src.corpus.preprocess window \
		--input $(DATA_DIR)/processed/segmented \
		--output $(DATA_DIR)/processed/windowed \
		--window-size 1000 \
		--overlap 500

preprocess-normalize:
	@echo "$(YELLOW)Normalizing to narrative time...$(NC)"
	$(VENV)/bin/python -m src.corpus.preprocess normalize \
		--input $(DATA_DIR)/processed/windowed \
		--output $(DATA_DIR)/processed/normalized

#------------------------------------------------------------------------------
# FUNCTOR EXTRACTION
#------------------------------------------------------------------------------
extract-trajectories: extract-sentiment extract-arousal extract-epistemic extract-thematic extract-entropy
	@echo "$(GREEN)✓ All trajectories extracted$(NC)"

extract-sentiment:
	@echo "$(YELLOW)Extracting F_sentiment trajectories...$(NC)"
	$(VENV)/bin/python -m src.functors.sentiment \
		--input $(DATA_DIR)/processed/normalized \
		--output $(RESULTS_DIR)/trajectories/sentiment

extract-arousal:
	@echo "$(YELLOW)Extracting F_arousal trajectories...$(NC)"
	$(VENV)/bin/python -m src.functors.arousal \
		--input $(DATA_DIR)/processed/normalized \
		--output $(RESULTS_DIR)/trajectories/arousal

extract-epistemic:
	@echo "$(YELLOW)Extracting F_epistemic trajectories...$(NC)"
	$(VENV)/bin/python -m src.functors.epistemic \
		--input $(DATA_DIR)/processed/normalized \
		--output $(RESULTS_DIR)/trajectories/epistemic

extract-thematic:
	@echo "$(YELLOW)Extracting F_thematic trajectories...$(NC)"
	$(VENV)/bin/python -m src.functors.thematic \
		--input $(DATA_DIR)/processed/normalized \
		--output $(RESULTS_DIR)/trajectories/thematic

extract-entropy:
	@echo "$(YELLOW)Extracting F_entropy (Shannon) trajectories...$(NC)"
	$(VENV)/bin/python -m src.functors.entropy \
		--input $(DATA_DIR)/processed/normalized \
		--output $(RESULTS_DIR)/trajectories/entropy \
		--compute-jsd

#------------------------------------------------------------------------------
# STRUCTURAL DETECTION
#------------------------------------------------------------------------------
detect-all: detect-harmon detect-kishotenketsu
	@echo "$(GREEN)✓ All structural detection complete$(NC)"

detect-harmon:
	@echo "$(YELLOW)Detecting Harmon Circle structure...$(NC)"
	$(VENV)/bin/python -m src.detectors.harmon \
		--input $(DATA_DIR)/processed/normalized \
		--output $(RESULTS_DIR)/structure/harmon

detect-kishotenketsu:
	@echo "$(YELLOW)Detecting kishōtenketsu structure...$(NC)"
	$(VENV)/bin/python -m src.detectors.kishotenketsu \
		--input $(DATA_DIR)/processed/normalized \
		--trajectories $(RESULTS_DIR)/trajectories/epistemic \
		--output $(RESULTS_DIR)/structure/kishotenketsu

#------------------------------------------------------------------------------
# CLUSTERING & ANALYSIS
#------------------------------------------------------------------------------
cluster:
	@echo "$(YELLOW)Clustering trajectories...$(NC)"
	$(VENV)/bin/python -m src.clustering.shape_cluster \
		--input $(RESULTS_DIR)/trajectories \
		--output $(RESULTS_DIR)/clusters \
		--n-clusters $(N_CLUSTERS) \
		--method hierarchical

analyze-all: analyze-transfer analyze-temporal analyze-editorial analyze-functor-stability
	@echo "$(GREEN)✓ All analyses complete$(NC)"

analyze-transfer:
	@echo "$(YELLOW)Running cross-cultural transfer analysis...$(NC)"
	$(VENV)/bin/python -m src.clustering.transfer_analysis \
		--clusters $(RESULTS_DIR)/clusters \
		--output $(RESULTS_DIR)/analysis/transfer

analyze-temporal:
	@echo "$(YELLOW)Running temporal drift analysis...$(NC)"
	$(VENV)/bin/python -m src.clustering.temporal_analysis \
		--clusters $(RESULTS_DIR)/clusters \
		--output $(RESULTS_DIR)/analysis/temporal

analyze-editorial:
	@echo "$(YELLOW)Running editorial gatekeeping analysis...$(NC)"
	$(VENV)/bin/python -m src.clustering.editorial_analysis \
		--clusters $(RESULTS_DIR)/clusters \
		--output $(RESULTS_DIR)/analysis/editorial

analyze-functor-stability:
	@echo "$(YELLOW)Running functor stability analysis (ARI)...$(NC)"
	$(VENV)/bin/python -m src.clustering.functor_stability \
		--clusters $(RESULTS_DIR)/clusters \
		--output $(RESULTS_DIR)/analysis/functor_stability

#------------------------------------------------------------------------------
# EXTENDED ANALYSIS (Q8-Q11)
#------------------------------------------------------------------------------
analyze-extended: analyze-gutenberg-parenthesis analyze-dopamine analyze-gilgamesh analyze-entropy
	@echo "$(GREEN)✓ All extended analyses complete$(NC)"

analyze-gutenberg-parenthesis:
	@echo "$(YELLOW)Q8: Testing Gutenberg Parenthesis hypothesis...$(NC)"
	$(VENV)/bin/python -m src.clustering.gutenberg_parenthesis \
		--ancient $(DATA_DIR)/raw/ancient \
		--print $(DATA_DIR)/raw/gutenberg \
		--digital $(DATA_DIR)/raw/streaming \
		--trajectories $(RESULTS_DIR)/trajectories \
		--output $(RESULTS_DIR)/analysis/gutenberg_parenthesis
	@echo "$(GREEN)✓ Gutenberg Parenthesis analysis complete$(NC)"

analyze-dopamine:
	@echo "$(YELLOW)Q9: Running dopamine optimization analysis...$(NC)"
	$(VENV)/bin/python -m src.clustering.dopamine_analysis \
		--streaming $(DATA_DIR)/raw/streaming \
		--trajectories $(RESULTS_DIR)/trajectories \
		--output $(RESULTS_DIR)/analysis/dopamine
	@echo "$(GREEN)✓ Dopamine analysis complete$(NC)"

analyze-gilgamesh:
	@echo "$(YELLOW)Q10: Analyzing ancient epic structures...$(NC)"
	$(VENV)/bin/python -m src.clustering.gilgamesh_test \
		--ancient $(DATA_DIR)/raw/ancient \
		--trajectories $(RESULTS_DIR)/trajectories \
		--output $(RESULTS_DIR)/analysis/gilgamesh
	@echo "$(GREEN)✓ Gilgamesh test complete$(NC)"

analyze-entropy:
	@echo "$(YELLOW)Q11: Running Shannon entropy pattern analysis...$(NC)"
	$(VENV)/bin/python -m src.clustering.entropy_analysis \
		--entropy $(RESULTS_DIR)/trajectories/entropy \
		--clusters $(RESULTS_DIR)/clusters \
		--output $(RESULTS_DIR)/analysis/entropy_patterns
	@echo "$(GREEN)✓ Entropy analysis complete$(NC)"

#------------------------------------------------------------------------------
# VISUALIZATION & REPORTING
#------------------------------------------------------------------------------
visualize:
	@echo "$(YELLOW)Generating visualizations...$(NC)"
	$(VENV)/bin/python -m src.utils.visualize \
		--results $(RESULTS_DIR) \
		--output $(RESULTS_DIR)/figures
	@echo "$(GREEN)✓ Visualizations generated$(NC)"

report:
	@echo "$(YELLOW)Generating research report...$(NC)"
	$(VENV)/bin/python -m src.utils.report \
		--results $(RESULTS_DIR) \
		--output $(RESULTS_DIR)/report.html
	@echo "$(GREEN)✓ Report generated at $(RESULTS_DIR)/report.html$(NC)"

export-results:
	@echo "$(YELLOW)Exporting results...$(NC)"
	$(VENV)/bin/python -m src.utils.export \
		--results $(RESULTS_DIR) \
		--format csv,json \
		--output $(RESULTS_DIR)/exports
	@echo "$(GREEN)✓ Results exported$(NC)"

#------------------------------------------------------------------------------
# FULL PIPELINES
#------------------------------------------------------------------------------
all: setup corpus-all preprocess extract-trajectories detect-all cluster analyze-all visualize report
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Full research pipeline complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

replication: clean setup corpus-all preprocess extract-trajectories detect-all cluster analyze-all visualize report export-results
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Full replication complete!$(NC)"
	@echo "$(GREEN)========================================$(NC)"

# Reagan et al. replication specifically
replicate-reagan:
	@echo "$(YELLOW)Replicating Reagan et al. (2016)...$(NC)"
	$(VENV)/bin/python -m src.replication.reagan \
		--corpus $(DATA_DIR)/raw/gutenberg \
		--output $(RESULTS_DIR)/replication/reagan
	@echo "$(GREEN)✓ Reagan replication complete$(NC)"

#------------------------------------------------------------------------------
# UTILITIES
#------------------------------------------------------------------------------
clean:
	rm -rf $(RESULTS_DIR)/*
	rm -rf $(DATA_DIR)/processed/*
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned generated files$(NC)"

clean-data:
	rm -rf $(DATA_DIR)/raw/*
	rm -rf $(DATA_DIR)/processed/*
	@echo "$(GREEN)✓ Cleaned all data$(NC)"

clean-all: clean clean-data
	rm -rf $(VENV)
	@echo "$(GREEN)✓ Cleaned everything including virtual environment$(NC)"

notebook:
	$(VENV)/bin/jupyter lab --notebook-dir=$(NOTEBOOKS_DIR)

validate:
	@echo "$(YELLOW)Validating data integrity...$(NC)"
	$(VENV)/bin/python -m src.utils.validate \
		--data $(DATA_DIR) \
		--results $(RESULTS_DIR)
	@echo "$(GREEN)✓ Validation complete$(NC)"

# Show current status of the research
status:
	@echo "$(YELLOW)Research Pipeline Status$(NC)"
	@echo "========================="
	@echo ""
	@echo "Raw Data:"
	@ls -la $(DATA_DIR)/raw 2>/dev/null || echo "  (no raw data)"
	@echo ""
	@echo "Processed Data:"
	@ls -la $(DATA_DIR)/processed 2>/dev/null || echo "  (no processed data)"
	@echo ""
	@echo "Results:"
	@ls -la $(RESULTS_DIR) 2>/dev/null || echo "  (no results)"
