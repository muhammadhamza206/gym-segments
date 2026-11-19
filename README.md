# gym-segments

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python fit.py
```

`fit.py` prints:

- The training and test set sizes.
- For each k from 1 to 8, k, the training spread, and the test spread.
- How many training members are in each cluster.
- The cluster centers in the original feature units.
- For each segment, the member count, average monthly extras spend, how many members spend under $20, and the total monthly net if that whole segment got the credit.
- The baseline monthly net when every member gets the credit, the v1 monthly net when only cluster 0 gets the credit, and the difference.
- For a new member, the distance to each cluster center, the cluster, the segment name, and the credit decision.

`fit.py` saves `elbow.png`, with training spread and test spread plotted against k.
