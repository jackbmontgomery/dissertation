# Dissertation

## Notes
- I am happy with a comparison on the use of more exotic / gradient based sampling for the inverse problem.
- This is also a good way to try and get some other experiments into the mix and see how the method fairs in those experiments
- Some good content writing about what MCLMC etc.

# For using pmap on single CPU

import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
