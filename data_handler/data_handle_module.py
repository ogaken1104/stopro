import h5py
import jax.numpy as jnp


class HdfOperator:
    def __init__(self, simulation_path='..'):
        self.test_path = f'{simulation_path}/data_input/test.hdf5'
        self.train_path = f'{simulation_path}/data_input/train.hdf5'
        self.infer_path = f'{simulation_path}/data_output/infer.hdf5'
        self.record_path = f'{simulation_path}/data_output/other.hdf5'
        self.analysis_path = f'{simulation_path}/data_output/analysis.hdf5'
        self.analysis_text_path = f'{simulation_path}/data_output/analysis.txt'
        # self.lbls_train = ['r', 'f']
        # self.lbls_test = ['r', 'f']
        # self.lbls_infer = ['f', 'std']
        # self.lbls_record = ['x', 'fun']
        # self.vnames_train = ['ux', 'uy', 'p', 'fx', 'fy', 'divu']
        # self.vnames_test = ['ux', 'uy', 'p']
        # self.vnames_infer = ['ux', 'uy', 'p']
        # self.lbls_analysis = ['abs_error', 'rel_error',
        #                       'max_abs_error', 'max_rel_error',  'mean_abs_error', 'mean_rel_error']
        # self.vnames_analysis = ['ux', 'uy', 'p']

    def load_data(self, data_path, lbls, vnames):
        with h5py.File(data_path, 'r') as file:
            data = []
            for lbl in lbls:
                vals = []
                for vname in vnames:
                    vals.append(jnp.array(file[f'{lbl}/{vname}'][()]))
                data.append(vals)
        return data

    def save_data(self, data_path, lbls, vnames, vals_list):
        with h5py.File(data_path, 'a') as file:
            for lbl, vals in zip(lbls, vals_list):
                try:
                    dir_lbl = file.create_group(lbl)
                except:
                    del file[lbl]
                    dir_lbl = file.create_group(lbl)
                for vname, val in zip(vnames, vals):
                    dir_lbl.create_dataset(vname, data=val)

    def load_test_data(self, lbls_test, vnames_test):
        return self.load_data(self.test_path, lbls_test, vnames_test)

    def load_train_data(self, lbls_train, vnames_train):
        return self.load_data(self.train_path, lbls_train, vnames_train)

    def load_infer_data(self, lbls_infer, vnames_infer):
        return self.load_data(self.infer_path, lbls_infer, vnames_infer)

    def load_analysis_data(self, lbls_analysis, vnames_analysis):
        return self.load_data(self.analysis_path, lbls_analysis, vnames_analysis)

    def save_test_data(self, lbls_test, vnames_test, vals_list):
        self.save_data(self.test_path, lbls_test,
                       vnames_test, vals_list)

    def save_train_data(self, lbls_train, vnames_train, vals_list):
        self.save_data(self.train_path, lbls_train,
                       vnames_train, vals_list)

    def save_infer_data(self, lbls_infer, vnames_infer, vals_list):
        self.save_data(self.infer_path, lbls_infer,
                       vnames_infer, vals_list)

    def save_analysis_data(self, lbls_analysis, vnames_analysis, vals_list):
        self.save_data(self.analysis_path, lbls_analysis,
                       vnames_analysis, vals_list,)

    def save_record(self, lbls_record, records):
        with h5py.File(self.record_path, 'a') as file:
            for lbl, record in zip(lbls_record, records):
                try:
                    file.create_dataset(lbl, data=record)
                except:
                    del file[lbl]
                    file.create_dataset(lbl, data=record)
                    print(f'{lbl} was renewed')

    def load_record(self, lbls_record):
        with h5py.File(self.record_path, 'a') as file:
            record = file[f'{lbls_record}'][()]
        return record


def printname(name):
    print(name)


def printname_only_dir(name):
    if not '/' in name:
        print(name)
