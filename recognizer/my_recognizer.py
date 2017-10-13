import warnings
from asl_data import SinglesData
import math


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    for index in range(len(test_set.get_all_sequences())):
      Xlengths = test_set.get_all_Xlengths()
      q_X, q_lengths = Xlengths[index]
      query_dict = {}
      best = -math.inf
      best_guess = None
      for word in models:
        try:
          logl = models[word].score(q_X, q_lengths)
          query_dict[word] = logl
          if logl > best:
            best = logl
            best_guess = word
        except:
          query_dict[word] = None
      probabilities.append(query_dict)
      guesses.append(best_guess)
    return probabilities, guesses

