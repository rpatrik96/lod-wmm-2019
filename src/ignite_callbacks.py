from os.path import isfile, join, abspath, dirname

import numpy as np
import pandas as pd

def early_stopping_score(evaluator):
    return -evaluator.state.metrics["loss"]  # improvement = higher score

"""WMM"""

def wmm_callback(engine, model_list, wmm_object):
    if type(model_list) is not list:
        wmm_object.run(model_list)

    else:
        # for models which consist of building blocks
        for model in model_list:
            wmm_object.run(model)

"""-------------------------------------------------------------------------"""
"""------------------------IgniteCustomLogger-------------------------------"""
"""-------------------------------------------------------------------------"""

class IgniteCustomLogger(object):
    @staticmethod
    def log_init(engine, interval):
        engine.state.num_steps = 0
        engine.state.interval = interval
        engine.state.time_series = {}

        engine.state.time_series["train_error"] = []
        engine.state.time_series["valid_error"] = []

    @staticmethod
    def log_validate(engine, evaluator, valid_loader, var_name):
        if not engine.state.num_steps % engine.state.interval:
            engine.state.time_series["train_error"].append(engine.state.output)

            evaluator.run(valid_loader)
            engine.state.time_series["valid_error"].append(evaluator.state.metrics["loss"])
            print("Validation loss: ", evaluator.state.metrics["loss"])
            # print("Acc: ", evaluator.state.metrics["acc"])

        engine.state.num_steps += 1

    @staticmethod
    def log_test(engine, tester, test_loader, model, model_params, wmm_object, df_path, log_path):

        tester.run(test_loader)
        print("Test loss: ", tester.state.metrics["loss"])

        # save time series
        # engine.state.time_series["entropy_dict"] = wmm_object.entropy_dict
        # np.save(log_path, engine.state.time_series)
        print("No time series saved!")

        # save results (merged from 2 dicts)
        temp_dict = {**wmm_object.__dict__, **model_params.__dict__, **tester.state.metrics}
        temp_dict['num_steps'] = engine.state.num_steps * engine.state.interval
        temp_dict['cumulative_entropy'] = wmm_object.calculate_entropy(model)
        del temp_dict['entropy_dict']
        del temp_dict['weight_dict']

        test_filename = 'test_reinit.tsv' if 'alpha' in temp_dict.keys() else 'test_shuffle.tsv'
        df_path = join(dirname(dirname(abspath(__file__))), test_filename)

        pd.DataFrame.from_dict(temp_dict).to_csv(
                df_path,
                sep='\t',
                index=False,
                header=True if not isfile(df_path) else False,
                mode='a')

    @staticmethod
    def lr_scheduling(engine, scheduler):
        # scheduler.step()
        pass
