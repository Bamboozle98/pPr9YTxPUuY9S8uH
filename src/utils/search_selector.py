from src.utils.GridSearch import grid_search_setup
from src.utils.RandomSearch import random_search_setup


def search_selection(model, selection):
    if selection == 'gs':
        search = grid_search_setup(model)
        return search
    if selection == 'rs':
        search = random_search_setup(model)
        return search
    else:
        print("Invalid Search Selection")
        return

