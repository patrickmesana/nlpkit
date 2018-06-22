"""
Utilities
---------

Reading or Writing files, spliting strings ...
"""
import os
import pickle
from pathlib import Path
from time import time
import json


def print_time_start(message="Starting..."):
    print(message)
    t0 = time()
    return t0


def print_time_stop(t0):
    print("done in %0.3fs." % (time() - t0))


def open_ext(path, ext, w=False):
    if ext == "json":
        if w:
            return open(path, "w", encoding="utf8")
        else:
            return open(path, "r", encoding="utf8")

    else:
        if w:
            return open(path, "wb")
        else:
            return open(path, "rb")


def dump_json(data, complete_path):
    with open_ext(complete_path, "json", w=True) as fout:
        json.dump(data, fout)


def do_or_load(path, do_fct, read_fct, write_fct, ext, should_overwrite=False, base_path="./input/session/"):
    complete_path = base_path + path + "." + ext
    output_file = Path(complete_path)
    does_file_exists = output_file.is_file()
    if does_file_exists and not should_overwrite:
        print('loading from cache : ' + complete_path)
        with open_ext(complete_path, ext, w=False) as fin:
            data = read_fct(fin)
    else:
        data = do_fct()
        write_file_and_create_folder_if_needed(base_path, complete_path, data, ext, write_fct)
    return data


def write_file_and_create_folder_if_needed(base_path, complete_path, data, ext, write_fct):
    create_folder_if_not_exists(base_path)
    with open_ext(complete_path, ext, w=True) as fout:
        write_fct(data, fout)


def create_folder_if_not_exists(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)


def file_to_lines(fin):
    return [line.strip() for line in fin]


def lines_to_file(lines, fout):
    return [fout.write(line) for line in lines]


def do_or_load_pickle(path, do_fct, should_overwrite=False):
    return do_or_load(path, do_fct, pickle.load, pickle.dump, "p", should_overwrite)


def do_or_load_json(path, do_fct, should_overwrite=False, base_path="./input/session/"):
    return do_or_load(path, do_fct, json.load, json.dump, "json", should_overwrite, base_path)


def do_or_load_lines(path, do_fct, should_overwrite=False):
    return do_or_load(path, do_fct, file_to_lines, lines_to_file, "txt", should_overwrite)


def comma_join(a_list):
    return ",".join(a_list)


def comma_split(a_string):
    return a_string.split(",")


def space_split(line):
    return line.split(" ")


def space_join(line_as_list):
    return " ".join(line_as_list)


def iterate_list_with_window(a_list, gram_as_list):
    window_size = len(gram_as_list)
    i = 0
    while i < len(a_list):
        if a_list[i:i + window_size] == gram_as_list:
            yield gram_as_list, list(range(i, i + window_size)), True
            i += window_size
        else:
            yield [a_list[i]], [i], False
            i += 1


def flatten(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def lazy(obj):
    return lambda: obj