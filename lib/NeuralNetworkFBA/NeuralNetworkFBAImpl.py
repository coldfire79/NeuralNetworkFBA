# -*- coding: utf-8 -*-
#BEGIN_HEADER
import logging
import os
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from installed_clients.DataFileUtilClient import DataFileUtil
from installed_clients.KBaseReportClient import KBaseReport
#END_HEADER


class NeuralNetworkFBA:
    '''
    Module Name:
    NeuralNetworkFBA

    Module Description:
    A KBase module: NeuralNetworkFBA
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = ""
    GIT_COMMIT_HASH = ""

    #BEGIN_CLASS_HEADER
    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']
        logging.basicConfig(format='%(created)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        #END_CONSTRUCTOR
        pass


    def run_NeuralNetworkFBA(self, ctx, params):
        """
        This example function accepts any number of parameters and returns results in a KBaseReport
        :param params: instance of mapping from String to unspecified object
        :returns: instance of type "ReportResults" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN run_NeuralNetworkFBA

        #######################################################################
        #  check out the input data for training
        #######################################################################
        print ("Input parameter", params['train_data'])
        dfu = DataFileUtil(self.callback_url)
        input_train = dfu.get_objects({'object_refs': [params['train_data']]})['data'][0]

        #print(input_train['data']['instances'])

        trainset = dict()
        for key in input_train['data']['instances']:
            idx = int(key)
            trainset[idx] = []
            for val in input_train['data']['instances'][key]:
                if val == '': trainset[idx].append(np.nan)
                else: trainset[idx].append(float(val))
        
        #print(trainset)

        tbl_cols = [info['attribute'] for info in input_train['data']['attributes']]
        train_df = pd.DataFrame.from_dict(trainset, orient='index', columns=tbl_cols).sort_index()

        print(tbl_cols)
        print(train_df)

        #######################################################################
        #  MLP training
        #######################################################################
        X = train_df.iloc[:,0:-1].values
        y = train_df.iloc[:,-1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=0.1, random_state=0)
        regr = MLPRegressor(random_state=0, max_iter=500).fit(X_train, y_train)
        regr.predict(X_test)
        test_r2 = regr.score(X_test, y_test)
        print("Test R2:", test_r2)
        #######################################################################
        #  KBase report
        #######################################################################
        report = KBaseReport(self.callback_url)
        report_info = report.create({'report': {'objects_created':[],
                                                'text_message': "Test R2: {}".format(test_r2)},
                                                'workspace_name': params['workspace_name']})
        output = {
            'report_name': report_info['name'],
            'report_ref': report_info['ref'],
        }
        #END run_NeuralNetworkFBA

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method run_NeuralNetworkFBA return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]
    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
