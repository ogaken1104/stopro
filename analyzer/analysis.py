import numpy as np


def max_rel_error(true, pred):
    index = np.where(true != 0.0)
    true2 = true[index]
    pred2 = pred[index]
    if np.all(true == 0.0):
        max_rel_error = 0.0
    else:
        max_rel_error = np.max(abs((true2 - pred2) / true2))
    return max_rel_error


def rel_error(true, pred):
    true_max = np.max(true)
    zero_threshold = (
        true_max * 1e-7
    )  # ignore the data that test value is smaller than 1e-7
    index = np.where(abs(true) > zero_threshold)
    if np.all(abs(true) <= zero_threshold):
        rel_error = 0.0
        return rel_error
    true2 = true[index]
    pred2 = pred[index]
    rel_error = abs((true2 - pred2) / true2)
    # print(rel_error)
    return rel_error


def calc_abs_error(true, pred):
    """eliminate values outside the domain"""
    index_to_calculate = true != 0
    return np.abs(true - pred)[index_to_calculate]


def analyze_result(
    f_test,
    f_infer,
    std,
    theta,
    loss,
    norm_of_grads_list,
    lbls_kernel_arg,
    vnames_analysis,
):
    analysis_text_path = f"../data_output/analysis.txt"

    # calculate absolute and relative errors
    absolute_error = []
    relative_error = []
    max_abs_error = []
    max_rel_error = []
    mean_abs_error = []
    mean_rel_error = []
    for f_te, f_inf in zip(f_test, f_infer):
        abs_err = calc_abs_error(f_te, f_inf)
        rel_err = rel_error(f_te, f_inf)
        absolute_error.append(np.abs(f_te - f_inf))  # plot用については、領域外の値も削除せず保存する
        relative_error.append(rel_err)
        if np.any(abs_err):
            max_abs_error.append(np.max(abs_err))
            mean_abs_error.append(np.mean(abs_err))
        if np.any(rel_err):
            max_rel_error.append(np.max(rel_err))
            mean_rel_error.append(np.mean(rel_err[rel_err != 0.0]))

    # textファイルに保存
    with open(analysis_text_path, "w") as f:
        f.write(f"absolute_error\n")
        f.write(f"- max\n")
        for vname, abs_err in zip(vnames_analysis, max_abs_error):
            f.write(f"{vname} : {abs_err:.7f}\n")
        f.write("- mean\n")
        for vname, err in zip(vnames_analysis, mean_abs_error):
            f.write(f"{vname} : {err:.7f}\n")
        f.write(f"\nrelative_error\n")
        f.write(f"- max\n")
        for vname, rel_err in zip(vnames_analysis, max_rel_error):
            f.write(f"{vname} : {rel_err:.5f}\n")
        f.write("- mean\n")
        for vname, err in zip(vnames_analysis, mean_rel_error):
            f.write(f"{vname} : {err:.7f}\n")
        f.write("\nstd\n")
        f.write(f"- max\n")
        for vname, st in zip(vnames_analysis, std):
            f.write(f"{vname} : {np.max(st):.7f}\n")
        f.write(f"- mean\n")
        for vname, st in zip(vnames_analysis, std):
            f.write(f"{vname} : {np.mean(st):.7f}\n")
        f.write(f"\n- loss\n{loss[-1]:.5f}\n")
        f.write("\n- thata\n")
        print(theta[-1])
        if len(theta[-1]) % len(lbls_kernel_arg) != 0:
            f.write(f"noise ")
            np.savetxt(f, [theta[-1][-1:]], fmt="%.3f")
            theta[-1] = theta[-1][:-1]
        for i, thet in enumerate(np.split(theta[-1], len(lbls_kernel_arg))):
            f.write(f"{lbls_kernel_arg[i]: <5}")
            np.savetxt(f, [thet], fmt="%.3f")
        f.write(f"\n- final norm of grads\n")
        try:
            f.write(f"{norm_of_grads_list[-1]}")
        except:
            pass
    # hdf5ファイルに保存
    vals_list = [
        absolute_error,
        relative_error,
        max_abs_error,
        max_rel_error,
        mean_abs_error,
        mean_rel_error,
    ]

    return vals_list


if __name__ == "__main__":
    analyze_result()
