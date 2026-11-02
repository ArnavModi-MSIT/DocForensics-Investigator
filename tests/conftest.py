import os, sys
os.environ.setdefault("DOCFOR_EMBED", "off")          # unit tests must not depend on the embedding model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
