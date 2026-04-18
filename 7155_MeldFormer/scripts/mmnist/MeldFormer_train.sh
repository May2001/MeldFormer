SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../.."
export CUDA_VISIBLE_DEVICES=0
python tools/train.py  -d mmnist  -c configs/mmnist/Meldformer.py  --ex_name  work_dirs/meldformer_mmnist  -e 2000  --lr 5e-4