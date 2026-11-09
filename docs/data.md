# Data notes

The private WPP file was not available in this checkout. Its shape comes from the previous code: a member ID, merchant ID, event time, anonymous flag, merchant name, category IDs, and sometimes a session ID. The old EDA code also referred to partner, paid level, and year. Those fields have no direct match in Retailrocket.

Retailrocket gives us visitor IDs, item IDs, timestamps, event types, and category properties. In the app, a visitor becomes a user and an item stands in for a merchant. This makes the recommendation task similar enough to test the code in public. It does not make retail products equivalent to WPP merchants or reproduce WPP business reports.

The loader freezes category properties at the training cutoff. A product with no category property still enters the catalog if it appeared in training; a later product can enter only if its metadata was already published. For the WPP file, names and categories are taken from pre-cutoff events because there is no separate catalog snapshot.

Retailrocket's [dataset page and license](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) describe the event files and the noncommercial Creative Commons terms. Other public options have tradeoffs: [Dressipi's fashion sessions](https://www.recsyschallenge.com/2022/) fit a session task but its [terms](https://www.recsyschallenge.com/2022/Dressipi2022_TC.pdf) restrict reuse; [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) has user, item, time, and metadata but reviews are a different behavior and its maintainers [do not assign a dataset license](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/discussions/1).

No dataset here measures the effect of a recommendation being shown. Offline ranking metrics cannot establish reward, long-term satisfaction, or how a production system would behave under live traffic.
