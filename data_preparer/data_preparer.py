import os
import shutil

import yaml

from stopro.data_handler.data_handle_module import HdfOperator


class DataPreparer:
    def __init__(self, project_name, simulation_name, class_data_generator) -> None:
        self.simulation_path = f"{project_name}/{simulation_name}"
        self.class_data_generator = class_data_generator
        if self.class_data_generator.__name__ in ["Sinusoidal", "SinusoidalCylinder"]:
            self.update_params = self._update_params_sinusoidal
        elif self.class_data_generator.__name__ == "Sin1D":
            self.update_params = self._update_params_sin_1D
        else:
            # update_paramsが定義されていない場合は何もしない
            self.update_params = lambda: None

    def create_directory(self):
        # Make simulation directory
        if os.path.exists(self.simulation_path):
            shutil.rmtree(self.simulation_path)
        data_input_path = self.simulation_path + "/data_input"
        data_output_path = self.simulation_path + "/data_output"
        os.makedirs(data_input_path)
        os.makedirs(data_output_path)
        os.makedirs(self.simulation_path + "/fig")
        os.makedirs(self.simulation_path + "/scripts")

    def load_params(self, system_name="sinusoidal", use_existing_params=False):
        if use_existing_params:
            params_path = use_existing_params
        else:
            params_path = (
                f'{os.environ["HOME"]}/opt/stopro/default_params/{system_name}'
            )
        with open(f"{params_path}/params_prepare.yaml") as file:
            params_prepare = yaml.safe_load(file)
        with open(f"{params_path}/params_main.yaml") as file:
            params_main = yaml.safe_load(file)
        with open(
            f'{os.environ["HOME"]}/opt/stopro/default_params/common/lbls.yaml'
        ) as file:
            self.lbls = yaml.safe_load(file)
        self.vnames = params_prepare["vnames"]
        self.params_setting = params_prepare["setting"]
        self.params_generate_test = params_prepare["generate_test"]
        self.params_generate_training = params_prepare["generate_training"]
        self.params_kernel_arg = params_prepare["kernel_arg"]
        self.params_plot = params_prepare["plot"]
        self.params_num_points = {}
        self.params_main = params_main

    def make_data(
        self,
        plot_training=True,
        plot_test=False,
        save_data=True,
        save_plot=True,
        return_data=False,
    ):
        # DataGeneratorのインスタンスを作成
        # 訓練データを生成して保存
        sample = self.class_data_generator(**self.params_setting)
        r_train, f_train = sample.generate_training_data(
            **self.params_generate_training
        )
        # テストデータを生成して保存
        r_test, f_test = sample.generate_test(**self.params_generate_test)
        if save_data:
            hdf_operator = HdfOperator(self.simulation_path)
            hdf_operator.save_train_data(
                self.lbls["train"], self.vnames["train"], [r_train, f_train]
            )
            hdf_operator.save_test_data(
                self.lbls["test"], self.vnames["test"], [r_test, f_test]
            )
        # plot
        if plot_training or save_plot:
            sample.plot_train(
                save=save_plot, path=self.simulation_path, show=plot_training
            )
        if plot_test or save_plot:
            sample.plot_test(
                save=save_plot,
                val_limits=self.params_plot["val_limits"],
                path=self.simulation_path,
                show=plot_test,
            )
        # 生成された点の数を保存
        train_num = [len(_r) for _r in r_train]
        test_num = [len(_r) for _r in r_test]
        self.save_num(train_num, test_num)
        if return_data:
            return r_train, f_train, r_test, f_test

    def save_num(self, train_num, test_num):
        self.params_num_points["training"] = {
            self.vnames["train"][i]: train_num[i] for i in range(len(train_num))
        }
        self.params_num_points["training"]["sum"] = sum(
            train_num[i] for i in range(len(train_num))
        )
        self.params_num_points["test"] = {
            self.vnames["test"][i]: test_num[i] for i in range(len(test_num))
        }
        self.params_num_points["test"]["sum"] = sum(
            test_num[i] for i in range(len(test_num))
        )

    def save_params_prepare(self):
        params_prepare = {
            "vnames": self.vnames,
            "setting": self.params_setting,
            "generate_test": self.params_generate_test,
            "generate_training": self.params_generate_training,
            "kernel_arg": self.params_kernel_arg,
            "plot": self.params_plot,
            "num_points": self.params_num_points,
        }
        with open(f"{self.simulation_path}/data_input/params_prepare.yaml", "w") as f:
            yaml.dump(params_prepare, f)

    def save_params_main(self):
        with open(f"{self.simulation_path}/data_input/params_main.yaml", "w") as f:
            yaml.dump(self.params_main, f)

    def save_lbls(self):
        with open(f"{self.simulation_path}/data_input/lbls.yaml", "w") as f:
            yaml.dump(self.lbls, f)

    def _update_params_sinusoidal(self):
        if self.params_setting["infer_gradp"]:
            self.vnames["test"] = ["ux", "uy", "px", "py"]
            self.vnames["infer"] = ["ux", "uy", "px", "py"]
            self.vnames["analysis"] = ["ux", "uy", "px", "py"]
        if self.params_setting["use_difu"]:
            self.params_kernel_arg = ["uxux", "uyuy", "pp", "uxuy", "uxp", "uyp"]
            if self.params_generate_training["without_f"]:
                self.vnames["train"] = ["ux", "uy", "difux", "difuy", "divu", "difp"]
            elif self.params_setting["use_diff"]:
                self.vnames["train"] = [
                    "ux",
                    "uy",
                    "difux",
                    "difuy",
                    "fx",
                    "fy",
                    "diffx",
                    "diffy",
                    "divu",
                    "difp",
                ]
            elif self.params_setting["use_difp"]:
                self.vnames["train"] = [
                    "ux",
                    "uy",
                    "difux",
                    "difuy",
                    "fx",
                    "fy",
                    "divu",
                    "difp",
                ]
            else:
                self.vnames["train"] = [
                    "ux",
                    "uy",
                    "difux",
                    "difuy",
                    "fx",
                    "fy",
                    "divu",
                ]
        if self.params_generate_test["infer_governing_eqs"]:
            self.vnames["analysis"] = ["fx", "fy", "divu"]
            self.vnames["infer"] = ["fx", "fy", "divu"]
            self.vnames["test"] = ["fx", "fy", "divu"]
        if self.params_generate_test["infer_du_boundary"]:
            self.vnames["analysis"] = ["duxx", "duxy", "duyx", "duyy", "p"]
            self.vnames["infer"] = ["duxx", "duxy", "duyx", "duyy", "p"]
            self.vnames["test"] = ["duxx", "duxy", "duyx", "duyy", "p"]
        if self.params_generate_test["infer_du_grid"]:
            val = "duxy"
            self.vnames["analysis"] = [val]
            self.vnames["infer"] = [val]
            self.vnames["test"] = [val]
        if self.params_setting["infer_difp"]:
            val = "difp"
            self.vnames["analysis"] = [val]
            self.vnames["infer"] = [val]
            self.vnames["test"] = [val]

    def _update_params_sin_1D(self):
        if self.params_setting["use_pbc_points"]:
            self.vnames["train"] = ["y", "pbc_y", "ly"]
