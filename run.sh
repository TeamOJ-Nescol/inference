source darts/bin/activate
export PYTHONPATH="$PYTHONPATH:$(pwd)/src/rf-detr"
fastapi dev src/server.py
