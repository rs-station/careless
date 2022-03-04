name = "Crossvalidation"
description = """
Careless supports two sorts of crossvalidation. The first is used to assess the validity of a particular merging
strategy (see Model Selection). The second is meant to assess the quality of a data set (see Data Consistency). 
By default, careless will run neither. However, You can use any combination of these strategies using the provided flags. \n

  Model Selection\n
  ---------------\n
  Use the `--test-fraction` option to assess model fit and diagnose overfitting. This method will reserve a fraction of the 
  data into a test set, and output the predictions for each reflection observation in the training and test sets at the end
  of merging. \n

  Data Consistency\n
  ----------------\n
  Careless is able to provide half dataset merging results for assessing data resolution and quality. Setting the 
  `--merge-half-datasets` flag will first train the model on the full training set of data. Afterwards, the data will
  be split into two halves by randomly partitioning the images. With the model weights frozen, structure factors 
  are estimated by optimizing the loss function for each half dataset. The half dataset merging results will be written
  to a pair of files which can be analyzed to estimate conventional statistics such as CChalf.

"""


args_and_kwargs = (
    (
        ("--test-fraction",),
        {
            "help": "Output model predictions for a held-out fraction of data. This should be used for model selection purposes. "
            "By default, no data will be held out during training. ",
            "type": float,
            "default": None,
        },
    ),
    (
        ("--merge-half-datasets",),
        {
            "help": "After training, split the data in half randomly by image and merge each half using the scaling model learned on the training fraction. "
            "The output of the halves will be written to a file which can be used to estimate traditional CChalf type measures. The full data set will"
            " always be used to generate half data sets irrespective of the test fraction. ",
            "action": "store_true",
            "default": False,
        },
    ),
    (
        ("--half-dataset-repeats",),
        {
            "help": "Number of times to Repeat the half dataset crossvalidation. By default this is one.",
            "type": int,
            "default": 1,
        },
    ),
)
