python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no no no


# Layer Norm
python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no no layer

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no layer layer

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no layer no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms layer layer layer

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms layer layer no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms layer no no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms layer no layer

# Batch Norm
python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no no batch

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no batch no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no batch batch

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms batch batch batch

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms batch batch no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms batch no no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms batch no batch

# Mixed
python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no layer batch

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms no batch layer

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms layer no batch

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms batch no layer

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms layer batch no

python src/train.py --root ../../data/SelfGNN/pyg/ --name Computers --norms batch layer no



