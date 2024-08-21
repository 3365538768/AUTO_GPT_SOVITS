import os, sys

if len(sys.argv) == 1: sys.argv.append('v2')
# version = "v1" if sys.argv[1] == "v1" else "v2"
version="v2"
os.environ["version"] = version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings

warnings.filterwarnings("ignore")
import json, yaml, torch, pdb, re, shutil
import platform
import psutil
import signal

torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if (os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if (name == "jieba.cache"): continue
        path = "%s/%s" % (tmp, name)
        delete = os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site

site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if (site_packages_roots == []): site_packages_roots = ["%s/runtime/Lib/site-packages" % now_dir]
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    "%s\n%s/tools\n%s/tools/damo_asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError:
            pass
from tools import my_utils
import traceback
import shutil
import pdb
from subprocess import Popen
import signal
from config import python_exec, infer_device, is_half, exp_root, webui_port_main, webui_port_infer_tts, webui_port_uvr5, \
    webui_port_subfix, is_share
from tools.i18n.i18n import I18nAuto, scan_language_list

language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else "auto"
os.environ["language"] = language
if language != 'auto':
    i18n = I18nAuto(language=language)
else:
    i18n = I18nAuto()
from scipy.io import wavfile
from tools.my_utils import load_audio
from multiprocessing import cpu_count
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
import gradio.analytics as analytics

analytics.version_check = lambda: None
import gradio as gr

n_cpu = cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords = {"10", "16", "20", "30", "40", "A2", "A3", "A4", "P4", "A50", "500", "A60", "70", "80", "90", "M4",
                   "T4", "TITAN", "L4", "4060", "H"}
set_gpu_numbers = set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper() for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024 + 0.4))
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    set_gpu_numbers.add(0)
    default_batch_size = int(psutil.virtual_memory().total / 1024 / 1024 / 1024 / 2)
gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers = str(sorted(list(set_gpu_numbers))[0])


def fix_gpu_number(input):  # 将越界的number强制改到界内
    try:
        if (int(input) not in set_gpu_numbers): return default_gpu_numbers
    except:
        return input
    return input


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","): output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


pretrained_sovits_name = ["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                          "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name = [
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]

pretrained_model_list = (
pretrained_sovits_name[-int(version[-1]) + 2], pretrained_sovits_name[-int(version[-1]) + 2].replace("s2G", "s2D"),
pretrained_gpt_name[-int(version[-1]) + 2], "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
"GPT_SoVITS/pretrained_models/chinese-hubert-base")

_ = ''
for i in pretrained_model_list:
    if os.path.exists(i):
        ...
    else:
        _ += f'\n    {i}'
if _:
    print("warning:", i18n('以下模型不存在:') + _)

_ = [[], []]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    else:
        _[0].append("")  ##没有下pretrained模型的，说不定他们是想自己从零训底模呢
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
    else:
        _[-1].append("")
pretrained_gpt_name, pretrained_sovits_name = _

SoVITS_weight_root = ["SoVITS_weights_v2", "SoVITS_weights"]
GPT_weight_root = ["GPT_weights_v2", "GPT_weights"]
for root in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [name for name in pretrained_sovits_name if name != ""]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [name for name in pretrained_gpt_name if name != ""]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()
for path in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(path, exist_ok=True)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {
        "choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


p_label = None
p_uvr5 = None
p_asr = None
p_denoise = None
p_tts_inference = None


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


system = platform.system()


def kill_process(pid):
    if (system == "Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def change_label(if_label, path_list):
    global p_label
    if (if_label == True and p_label == None):
        path_list = my_utils.clean_path(path_list)
        cmd = '"%s" tools/subfix_webui.py --load_list "%s" --webui_port %s --is_share %s' % (
        python_exec, path_list, webui_port_subfix, is_share)
        yield i18n("打标工具WebUI已开启")
        print(cmd)
        p_label = Popen(cmd, shell=True)
    elif (if_label == False and p_label != None):
        kill_process(p_label.pid)
        p_label = None
        yield i18n("打标工具WebUI已关闭")


def change_uvr5(if_uvr5):
    global p_uvr5
    if (if_uvr5 == True and p_uvr5 == None):
        cmd = '"%s" tools/uvr5/webui.py "%s" %s %s %s' % (python_exec, infer_device, is_half, webui_port_uvr5, is_share)
        yield i18n("UVR5已开启")
        print(cmd)
        p_uvr5 = Popen(cmd, shell=True)
    elif (if_uvr5 == False and p_uvr5 != None):
        kill_process(p_uvr5.pid)
        p_uvr5 = None
        yield i18n("UVR5已关闭")


def change_tts_inference(if_tts, bert_path, cnhubert_base_path, gpu_number, gpt_path, sovits_path):
    global p_tts_inference
    if (if_tts == True and p_tts_inference == None):
        os.environ["gpt_path"] = gpt_path if "/" in gpt_path else "%s/%s" % (GPT_weight_root, gpt_path)
        os.environ["sovits_path"] = sovits_path if "/" in sovits_path else "%s/%s" % (SoVITS_weight_root, sovits_path)
        os.environ["cnhubert_base_path"] = cnhubert_base_path
        os.environ["bert_path"] = bert_path
        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_number(gpu_number)
        os.environ["is_half"] = str(is_half)
        os.environ["infer_ttswebui"] = str(webui_port_infer_tts)
        os.environ["is_share"] = str(is_share)
        # cmd = '"%s" GPT_SoVITS/inference_webui.py "%s"' % (python_exec, language)
        yield i18n("TTS推理进程已开启")
        # print(cmd)
        # p_tts_inference = Popen(cmd, shell=True)
    # elif (if_tts == False and p_tts_inference != None):
    #     kill_process(p_tts_inference.pid)
    #     p_tts_inference = None
    #     yield i18n("TTS推理进程已关闭")


from tools.asr.config import asr_dict


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    global p_asr
    if (p_asr == None):
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        asr_opt_dir = my_utils.clean_path(asr_opt_dir)
        check_for_exists([asr_inp_dir])
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += f" -p {asr_precision}"
        output_file_name = os.path.basename(asr_inp_dir)
        output_folder = asr_opt_dir or "output/asr_opt"
        output_file_path = os.path.abspath(f'{output_folder}/{output_file_name}.list')
        yield "ASR任务开启：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                 "visible": True}, {
            "__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        yield f"ASR任务完成, 查看终端进行下一步", {"__type__": "update", "visible": True}, {"__type__": "update",
                                                                                            "visible": False}, {
            "__type__": "update", "value": output_file_path}, {"__type__": "update", "value": output_file_path}, {
            "__type__": "update", "value": asr_inp_dir}
    else:
        yield "已有正在进行的ASR任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}, {"__type__": "update"}, {"__type__": "update"}, {
            "__type__": "update"}
        # return None


def close_asr():
    global p_asr
    if (p_asr != None):
        kill_process(p_asr.pid)
        p_asr = None
    return "已终止ASR进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


def open_denoise(denoise_inp_dir, denoise_opt_dir):
    global p_denoise
    if (p_denoise == None):
        denoise_inp_dir = my_utils.clean_path(denoise_inp_dir)
        denoise_opt_dir = my_utils.clean_path(denoise_opt_dir)
        check_for_exists([denoise_inp_dir])
        cmd = '"%s" tools/cmd-denoise.py -i "%s" -o "%s" -p %s' % (
        python_exec, denoise_inp_dir, denoise_opt_dir, "float16" if is_half == True else "float32")

        yield "语音降噪任务开启：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                      "visible": True}, {
            "__type__": "update"}, {"__type__": "update"}
        print(cmd)
        p_denoise = Popen(cmd, shell=True)
        p_denoise.wait()
        p_denoise = None
        yield f"语音降噪任务完成, 查看终端进行下一步", {"__type__": "update", "visible": True}, {"__type__": "update",
                                                                                                 "visible": False}, {
            "__type__": "update", "value": denoise_opt_dir}, {"__type__": "update", "value": denoise_opt_dir}
    else:
        yield "已有正在进行的语音降噪任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}, {"__type__": "update"}, {"__type__": "update"}
        # return None


def close_denoise():
    global p_denoise
    if (p_denoise != None):
        kill_process(p_denoise.pid)
        p_denoise = None
    return "已终止语音降噪进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


p_train_SoVITS = None


def open1Ba(batch_size, total_epoch, exp_name, text_low_lr_rate, if_save_latest, if_save_every_weights,
            save_every_epoch, gpu_numbers1Ba, pretrained_s2G, pretrained_s2D):
    global p_train_SoVITS
    if (p_train_SoVITS == None):
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
        check_for_exists([s2_dir], is_train=True)
        if (is_half == False):
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["model"]["version"] = version
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root[-int(version[-1]) + 2]
        data["name"] = exp_name
        data["version"] = version
        tmp_config_path = "%s/tmp_s2.json" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        yield "SoVITS训练开始：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                    "visible": True}
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        yield "SoVITS训练完成", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1Ba():
    global p_train_SoVITS
    if (p_train_SoVITS != None):
        kill_process(p_train_SoVITS.pid)
        p_train_SoVITS = None
    return "已终止SoVITS训练", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


p_train_GPT = None


def open1Bb(batch_size, total_epoch, exp_name, if_dpo, if_save_latest, if_save_every_weights, save_every_epoch,
            gpu_numbers, pretrained_s1):
    global p_train_GPT
    if (p_train_GPT == None):
        with open(
                "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml") as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        check_for_exists([s1_dir], is_train=True)
        if (is_half == False):
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root[-int(version[-1]) + 2]
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1" % s1_dir
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"] = fix_gpu_numbers(gpu_numbers.replace("-", ","))
        os.environ["hz"] = "25hz"
        tmp_config_path = "%s/tmp_s1.yaml" % tmp
        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        yield "GPT训练开始：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        yield "GPT训练完成", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1Bb():
    global p_train_GPT
    if (p_train_GPT != None):
        kill_process(p_train_GPT.pid)
        p_train_GPT = None
    return "已终止GPT训练", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


ps_slice = []


def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_exists([inp])
    if (os.path.exists(inp) == False):
        yield "输入路径不存在", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}, {
            "__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield "输入路径存在但既不是文件也不是文件夹", {"__type__": "update", "visible": True}, {"__type__": "update",
                                                                                                "visible": False}, {
            "__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (
            python_exec, inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha,
            i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "切割执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}, {
            "__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield "切割结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}, {
            "__type__": "update", "value": opt_root}, {"__type__": "update", "value": opt_root}, {"__type__": "update",
                                                                                                  "value": opt_root}
    else:
        yield "已有正在进行的切割任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}, {"__type__": "update"}, {"__type__": "update"}, {
            "__type__": "update"}


def close_slice():
    global ps_slice
    if (ps_slice != []):
        for p_slice in ps_slice:
            try:
                kill_process(p_slice.pid)
            except:
                traceback.print_exc()
        ps_slice = []
    return "已终止所有切割进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


ps1a = []


def open1a(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir):
    global ps1a
    inp_text = my_utils.clean_path(inp_text) #用于路径中分隔符问题
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text, inp_wav_dir])
    if (ps1a == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir,
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    "is_half": str(is_half)
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)#Popen(cmd, shell=True) 执行 cmd 所表示的命令。
            ps1a.append(p)
        yield "文本进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1a:
            p.wait()
        opt = []
        for i_part in range(all_parts):
            txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
            with open(txt_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(txt_path)
        path_text = "%s/2-name2text.txt" % opt_dir
        with open(path_text, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1a = []
        if len("".join(opt)) > 0:
            yield "文本进程成功", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        else:
            yield "文本进程失败", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的文本任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1a():
    global ps1a
    if (ps1a != []):
        for p1a in ps1a:
            try:
                kill_process(p1a.pid)
            except:
                traceback.print_exc()
        ps1a = []
    return "已终止所有1a进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


ps1b = []


def open1b(inp_text, inp_wav_dir, exp_name, gpu_numbers, ssl_pretrained_dir):
    global ps1b
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text, inp_wav_dir])
    if (ps1b == []):
        config = {
            "inp_text": inp_text,
            "inp_wav_dir": inp_wav_dir,
            "exp_name": exp_name,
            "opt_dir": "%s/%s" % (exp_root, exp_name),
            "cnhubert_base_dir": ssl_pretrained_dir,
            "is_half": str(is_half)
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1b.append(p)
        yield "SSL提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps1b:
            p.wait()
        ps1b = []
        yield "SSL提取进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的SSL提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1b():
    global ps1b
    if (ps1b != []):
        for p1b in ps1b:
            try:
                kill_process(p1b.pid)
            except:
                traceback.print_exc()
        ps1b = []
    return "已终止所有1b进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


ps1c = []


def open1c(inp_text, exp_name, gpu_numbers, pretrained_s2G_path):
    global ps1c
    inp_text = my_utils.clean_path(inp_text)
    check_for_exists([inp_text])
    if (ps1c == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        config = {
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": "GPT_SoVITS/configs/s2.json",
            "is_half": str(is_half)
        }
        gpu_names = gpu_numbers.split("-")
        all_parts = len(gpu_names)
        for i_part in range(all_parts):
            config.update(
                {
                    "i_part": str(i_part),
                    "all_parts": str(all_parts),
                    "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                }
            )
            os.environ.update(config)
            cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
            print(cmd)
            p = Popen(cmd, shell=True)
            ps1c.append(p)
        yield "语义token提取进程执行中", {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                    "visible": True}
        for p in ps1c:
            p.wait()
        opt = ["item_name\tsemantic_audio"]
        path_semantic = "%s/6-name2semantic.tsv" % opt_dir
        for i_part in range(all_parts):
            semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
            with open(semantic_path, "r", encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(semantic_path)
        with open(path_semantic, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ps1c = []
        yield "语义token提取进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的语义token提取任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1c():
    global ps1c
    if (ps1c != []):
        for p1c in ps1c:
            try:
                kill_process(p1c.pid)
            except:
                traceback.print_exc()
        ps1c = []
    return "已终止所有语义token进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


#####inp_text,inp_wav_dir,exp_name,gpu_numbers1a,gpu_numbers1Ba,gpu_numbers1c,bert_pretrained_dir,cnhubert_base_dir,pretrained_s2G
ps1abc = []


def open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c, bert_pretrained_dir,
             ssl_pretrained_dir, pretrained_s2G_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text, inp_wav_dir])
    if (ps1abc == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if (os.path.exists(path_text) == False or (os.path.exists(path_text) == True and len(
                    open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2)):
                config = {  #全局变量，后面直接拿来用
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)  #分配gpu
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc: p.wait()

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
            yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                     "visible": True}
            for p in ps1abc: p.wait()
            yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if (os.path.exists(path_semantic) == False or (
                    os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31)):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                          "visible": True}
                for p in ps1abc: p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        except:
            traceback.print_exc()
            close1abc()
            yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc = []
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


def switch_version(version_):
    os.environ['version'] = version_
    global version
    version = version_
    if pretrained_sovits_name[-int(version[-1]) + 2] != '' and pretrained_gpt_name[-int(version[-1]) + 2] != '':
        ...
    else:
        gr.Warning(i18n(f'未下载{version.upper()}模型'))
    return {'__type__': 'update', 'value': pretrained_sovits_name[-int(version[-1]) + 2]}, {'__type__': 'update',
                                                                                            'value':
                                                                                                pretrained_sovits_name[
                                                                                                    -int(version[
                                                                                                             -1]) + 2].replace(
                                                                                                    "s2G", "s2D")}, {
        '__type__': 'update', 'value': pretrained_gpt_name[-int(version[-1]) + 2]}, {'__type__': 'update',
                                                                                     'value': pretrained_gpt_name[
                                                                                         -int(version[-1]) + 2]}, {
        '__type__': 'update', 'value': pretrained_sovits_name[-int(version[-1]) + 2]}


def check_for_exists(file_list=[], is_train=False): #提示2、3、4、5、6中哪些有缺失
    _ = []
    if is_train == True and file_list:
        file_list.append(os.path.join(file_list[0], '2-name2text.txt'))
        file_list.append(os.path.join(file_list[0], '3-bert'))
        file_list.append(os.path.join(file_list[0], '4-cnhubert'))
        file_list.append(os.path.join(file_list[0], '5-wav32k'))
        file_list.append(os.path.join(file_list[0], '6-name2semantic.tsv'))
    for file in file_list:
        if os.path.exists(file):
            pass
        else:
            _.append(file)
    if _:
        if is_train:
            for i in _:
                if i != '':
                    gr.Warning(i)
            gr.Warning(i18n('以下文件或文件夹不存在:'))
        else:
            if len(_) == 1:
                if _[0]:
                    gr.Warning(i)
                gr.Warning(i18n('文件或文件夹不存在:'))
            else:
                for i in _:
                    if i != '':
                        gr.Warning(i)
                gr.Warning(i18n('以下文件或文件夹不存在:'))

def get_name(string):
    return string.split('/')[-1]
def autorun(train_file_name, train, gpu, reference, text, text_language, prompt_text, prompt_language,show):
    reference_name = get_name(reference)
    slice_inp_path = "resources/train/" + train_file_name
    slice_opt_root = "resources/slice/" + train_file_name
    threshold = -34
    min_length = 4000
    min_interval = 300
    hop_size = 10
    max_sil_kept = 500
    _max = 0.9
    alpha = 0.25
    n_process = 4
    if show:
        print(f"Step1：切割音频\n"
              f"参数：\n"
              f"slice_inp_path:{slice_inp_path}\n"
              f"slice_opt_root:{slice_opt_root}\n"
              f"threshold:{threshold}\n"
              f"min_length:{min_length}\n"
              f"min_interval:{min_interval}\n"
              f"hop_size:{hop_size}\n"
              f"max_sil_kept:{max_sil_kept}\n"
              f"_max:{_max}\n"
              f"alpha:{alpha}\n"
              f"n_process:{n_process}")
    if train == True:
        slice_generator = open_slice(slice_inp_path, slice_opt_root, threshold, min_length, min_interval, hop_size,
                                     max_sil_kept, _max, alpha, n_process)
        for message, visible_update_1, visible_update_2 ,visible_update_3,visible_update_4,visible_update_5in in slice_generator:
            if show:
                print(message)
    if show:
        print("Step1：切割音频结束")


    asr_inp_dir = "resources/slice/" + train_file_name
    asr_opt_dir = "resources/asr/" + train_file_name
    asr_model = "达摩 ASR (中文)"
    asr_size = "large"
    asr_lang = "zh"
    asr_precision = "float32"
    if show:
        print(f"Step2：asr转换音频\n"
              f"参数 :\n"
              f"asr_inp_dir={asr_inp_dir}\n"
              f"asr_opt_dir={asr_opt_dir}\n"
              f"asr_model={asr_model}\n"
              f"asr_size={asr_size}\n"
              f"asr_lang={asr_lang}\n"
              f"asr_precision={asr_precision}")

    if train == True:
        asr_generator = open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_size, asr_lang,asr_precision)
        for message, visible_update_1, visible_update_2 ,visible_update_3,visible_update_4,visible_update_5 in asr_generator:
            if show:
                print(message)
    if show:
        print("Step2:asr转换音频结束")

    inp_text = "resources/asr/" + train_file_name + "/" + train_file_name + ".list"
    inp_wav_dir = "resources/slice/" + train_file_name
    exp_name = "exp_" + train_file_name
    gpu_numbers1a = gpu
    gpu_numbers1Ba = gpu
    gpu_numbers1c = gpu
    bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    cnhubert_base_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    pretrained_s2G = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    if show:
        print("Step3:数据预处理一件三连开始\n")
        print("参数：\n")
        print(f"inp_text:{inp_text}")
        print(f"inp_wav_dir:{inp_wav_dir}")
        print(f"exp_name:{exp_name}")
        print(f"gpu_numbers1a:{gpu_numbers1a}")
        print(f"gpu_numbers1Ba:{gpu_numbers1Ba}")
        print(f"gpu_numbers1c:{gpu_numbers1c}")
        print(f"bert_pretrained_dir:{bert_pretrained_dir}")
        print(f"cnhubert_base_dir:{cnhubert_base_dir}")
        print(f"pretrained_s2G:{pretrained_s2G}")
    if train == True:
        open1abc_generator = open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c,
                                      bert_pretrained_dir, cnhubert_base_dir, pretrained_s2G)
        for message, visible_update_1, visible_update_2 in open1abc_generator:
            if show:
                print(message)
    if show:
        print("Step3:数据预处理一件三连结束\n")

    batch_size = 4
    total_epoch = 8
    exp_name = "exp_" + train_file_name
    text_low_lr_rate = 0.4
    if_save_latest = True
    if_save_every_weights = True
    save_every_epoch = 4
    gpu_numbers1Ba = gpu
    pretrained_s2G = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
    pretrained_s2D = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth"
    if show:
        print("Ste4：Sovits训练\n参数:\n")
        print(f"batch_size:{batch_size}\n"
              f"total_epoch:{total_epoch}\n"
              f"exp_name:{exp_name}\n"
              f"text_low_lr_rate:{text_low_lr_rate}\n"
              f"if_save_latest:{if_save_latest}\n"
              f"if_save_every_weights:{if_save_every_weights}\n"
              f"save_every_epoch:{save_every_epoch}\n"
              f"gpu_numbers1Ba:{gpu_numbers1Ba}\n"
              f"pretrained_s2G:{pretrained_s2G}\n"
              f"pretrained_s2D:{pretrained_s2D}")
    if train == True:
        open1Ba_generator = open1Ba(batch_size, total_epoch, exp_name, text_low_lr_rate, if_save_latest,
                                    if_save_every_weights, save_every_epoch, gpu_numbers1Ba, pretrained_s2G,
                                    pretrained_s2D)
        for message, visible_update_1, visible_update_2 in open1Ba_generator:
            if show:
                print(message)
    if show:
        print("Step4:Sovits训练结束\n")

    batch_size1Bb = 4
    total_epoch1Bb = 15
    exp_name = "exp_" + train_file_name
    if_dpo = False
    if_save_latest1Bb = True
    if_save_every_weights1Bb = True
    save_every_epoch1Bb = 5
    gpu_numbers1Bb = gpu
    pretrained_s1 = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
    if show:
        print("Ste5：GPT训练\n参数:\n")
        print(f"batch_size1Bb:{batch_size1Bb}\n"
              f"total_epoch1Bb:{total_epoch1Bb}\n"
              f"exp_name:{exp_name}\n"
              f"if_dpo:{if_dpo}\n"
              f"if_save_latest1Bb:{if_save_latest1Bb}\n"
              f"if_save_every_weights1Bb:{if_save_every_weights1Bb}\n"
              f"save_every_epoch1Bb:{save_every_epoch1Bb}\n"
              f"gpu_numbers1Bb:{gpu_numbers1Bb}\n"
              f"pretrained_s1:{pretrained_s1}")
    if train == True:
        open1Bb_generator = open1Bb(batch_size1Bb, total_epoch1Bb, exp_name, if_dpo, if_save_latest1Bb,
                                    if_save_every_weights1Bb, save_every_epoch1Bb, gpu_numbers1Bb, pretrained_s1)
        for message, visible_update_1, visible_update_2 in open1Bb_generator:
            if show:
                print(message)
    if show:
        print("Step5:GPT训练结束\n")
        print("Step6:打开推理\n参数:\n")
    GPT_dropdown = exp_name + "-e15.ckpt"
    gpu_number_1C = gpu
    SoVITS_dropdown = exp_name + "_e8_s96.pth"
    if_tts = True
    bert_pretrained_dir = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
    if show:
        print("打开推理参数：")
        print(f"GPT_dropdown={GPT_dropdown}\n"
              f"SoVITS_dropdown={SoVITS_dropdown}\n"
              f"bert_pretrained_dir={bert_pretrained_dir}\n"
              f"gpu_number_1C={gpu_number_1C}\n"
              )

    change_tts_inference_generator = change_tts_inference(if_tts, bert_pretrained_dir, cnhubert_base_dir, gpu_number_1C,
                                                           GPT_dropdown, SoVITS_dropdown)
    for message in change_tts_inference_generator:
        if show:
            print(message)
    if show:
        print("Step6:打开推理结束")

# train_file_name="shoulinrui.m4a"
# train=True
# gpu="0-0"
# reference="resources/slice/shoulinrui.m4a_0000063040_0000325440.wav"
# text="啊，一些重要的公司会以各种主题反复出现在商业就是这样的节目中，例如，啊，星巴克"
# text_language="zh"
# prompt_text="这是生成的结果"
# prompt_language="zh"
#
# autorun(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language)


import logging
import traceback

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("multipart.multipart").setLevel(logging.ERROR)
import LangSegment, os, re, sys, json
import pdb
import torch

version = os.environ.get("version", "v2")
pretrained_sovits_name = ["GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
                          "GPT_SoVITS/pretrained_models/s2G488k.pth"]
pretrained_gpt_name = [
    "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
    "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"]

_ = [[], []]
for i in range(2):
    if os.path.exists(pretrained_gpt_name[i]):
        _[0].append(pretrained_gpt_name[i])
    if os.path.exists(pretrained_sovits_name[i]):
        _[-1].append(pretrained_sovits_name[i])
pretrained_gpt_name, pretrained_sovits_name = _

if os.path.exists(f"./weight.json"):
    pass
else:
    with open(f"./weight.json", 'w', encoding="utf-8") as file:
        json.dump({'GPT': {}, 'SoVITS': {}}, file)

with open(f"./weight.json", 'r', encoding="utf-8") as file:
    weight_data = file.read()
    weight_data = json.loads(weight_data)
    gpt_path = os.environ.get(
        "gpt_path", weight_data.get('GPT', {}).get(version, pretrained_gpt_name))
    sovits_path = os.environ.get(
        "sovits_path", weight_data.get('SoVITS', {}).get(version, pretrained_sovits_name))
    if isinstance(gpt_path, list):
        gpt_path = gpt_path[0]
    if isinstance(sovits_path, list):
        sovits_path = sovits_path[0]

# gpt_path = os.environ.get(
#     "gpt_path", pretrained_gpt_name
# )
# sovits_path = os.environ.get("sovits_path", pretrained_sovits_name)
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
punctuation = set(['!', '?', '…', ',', '.', '-', " "])
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from tools.my_utils import load_audio
from tools.i18n.i18n import I18nAuto, scan_language_list

language = os.environ.get("language", "auto")
language = sys.argv[-1] if sys.argv[-1] in scan_language_list() else language
if language != 'auto':
    i18n = I18nAuto(language=language)
else:
    i18n = I18nAuto()

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

dict_language_v1 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}
dict_language_v2 = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("粤语"): "all_yue",  # 全部按中文识别
    i18n("韩文"): "all_ko",  # 全部按韩文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
    i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
    i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种
}
dict_language = dict_language_v1 if version == 'v1' else dict_language_v2

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path, prompt_language=None, text_language=None):
    global vq_model, hps, version, dict_language
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    version = hps.model.version
    # print("sovits版本:",hps.model.version)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    dict_language = dict_language_v1 if version == 'v1' else dict_language_v2
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["SoVITS"][version] = sovits_path
    with open("./weight.json", "w") as f:
        f.write(json.dumps(data))
    if prompt_language is not None and text_language is not None:
        if prompt_language in list(dict_language.keys()):
            prompt_text_update, prompt_language_update = {'__type__': 'update'}, {'__type__': 'update',
                                                                                  'value': prompt_language}
        else:
            prompt_text_update = {'__type__': 'update', 'value': ''}
            prompt_language_update = {'__type__': 'update', 'value': i18n("中文")}
        if text_language in list(dict_language.keys()):
            text_update, text_language_update = {'__type__': 'update'}, {'__type__': 'update', 'value': text_language}
        else:
            text_update = {'__type__': 'update', 'value': ''}
            text_language_update = {'__type__': 'update', 'value': i18n("中文")}
        return {'__type__': 'update', 'choices': list(dict_language.keys())}, {'__type__': 'update', 'choices': list(
            dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update




def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./weight.json") as f:
        data = f.read()
        data = json.loads(data)
        data["GPT"][version] = gpt_path
    with open("./weight.json", "w") as f: f.write(json.dumps(data))




def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx = audio.abs().max()
    if (maxx > 1): audio /= min(2, maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


dtype = torch.float16 if is_half == True else torch.float32


def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)  # .to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


from text import chinese


def get_phones_and_bert(text, language, version):
    if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日韩文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        if language == "zh":
            if re.search(r'[A-Za-z]', formattext):
                formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return get_phones_and_bert(formattext, "zh", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
                bert = get_bert_feature(norm_text, word2ph).to(device)
        elif language == "yue" and re.search(r'[A-Za-z]', formattext):
            formattext = re.sub(r'[a-z]', lambda x: x.group(0).upper(), formattext)
            formattext = chinese.mix_text_normalize(formattext)
            return get_phones_and_bert(formattext, "yue", version)
        else:
            phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        elif language == "auto_yue":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "zh":
                    tmp["lang"] = "yue"
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日韩文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones, bert.to(dtype), norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result


##ref_wav_path+prompt_text+prompt_language+text(单个)+text_language+top_k+top_p+temperature
# cache_tokens={}#暂未实现清理机制
cache = {}


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, how_to_cut=i18n("不切"), top_k=20,
                top_p=0.6, temperature=0.6, ref_free=False, speed=1, if_freeze=False, inp_refs=123):
    global cache
    if ref_wav_path:
        pass
    else:
        gr.Warning(i18n('请上传参考音频'))
    if text:
        pass
    else:
        gr.Warning(i18n('请填入推理文本'))
    t = []
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text

    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    if not ref_free:
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if (wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000):
                gr.Warning(i18n("参考音频在3~10秒范围外，请更换！"))
                raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if is_half == True:
                wav16k = wav16k.half().to(device)
                zero_wav_torch = zero_wav_torch.half().to(device)
            else:
                wav16k = wav16k.to(device)
                zero_wav_torch = zero_wav_torch.to(device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # .float()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)

    for i_text, text in enumerate(texts):
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()
        # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
        # print(cache.keys(),if_freeze)
        if (i_text in cache and if_freeze == True):
            pred_semantic = cache[i_text]
        else:
            with torch.no_grad():
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids, #这里为什么直接连起来导进去？
                    all_phoneme_len,
                    None if ref_free else prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    early_stop_num=hz * max_sec,
                )
                pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
                cache[i_text] = pred_semantic
        t3 = ttime()
        refers = []
        if (inp_refs):
            for path in inp_refs:
                try:
                    refer = get_spepc(hps, path.name).to(dtype).to(device)
                    refers.append(refer)
                except:
                    traceback.print_exc()
        if (len(refers) == 0): refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
        audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refers,
                                 speed=speed).detach().cpu().numpy()[0, 0])
        max_audio = np.abs(audio).max()  # 简单防止16bit爆音
        if max_audio > 1: audio /= max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" %
          (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3]))
          )
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip("。").split("。")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


def cut4(inp):
    inp = inp.strip("\n")
    opts = ["%s" % item for item in inp.strip(".").split(".")]
    opts = [item for item in opts if not set(item).issubset(punctuation)]
    return "\n".join(opts)


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    inp = inp.strip("\n")
    punds = {',', '.', ';', '?', '!', '、', '，', '。', '？', '！', ';', '：', '…'}
    mergeitems = []
    items = []

    for i, char in enumerate(inp):
        if char in punds:
            if char == '.' and i > 0 and i < len(inp) - 1 and inp[i - 1].isdigit() and inp[i + 1].isdigit():
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    opt = [item for item in mergeitems if not set(item).issubset(punds)]
    return "\n".join(opt)


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def process_text(texts):
    _text = []
    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError(i18n("请输入有效文本"))
    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)
    return _text


def change_choices():
    SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {
        "choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


SoVITS_weight_root = ["SoVITS_weights_v2", "SoVITS_weights"]
GPT_weight_root = ["GPT_weights_v2", "GPT_weights"]
for path in SoVITS_weight_root + GPT_weight_root:
    os.makedirs(path, exist_ok=True)


def get_weights_names(GPT_weight_root, SoVITS_weight_root):
    SoVITS_names = [i for i in pretrained_sovits_name]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [i for i in pretrained_gpt_name]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names(GPT_weight_root, SoVITS_weight_root)


def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def html_left(text, label='p'):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


def autorun2(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language,gpt_path,sovits_path,show):
    if not show:
        logging.basicConfig(level=logging.CRITICAL)
    exp_name = "exp_" + train_file_name
    reference_name = get_name(reference)
    cnhubert_base_dir = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
    gpt_path = gpt_path
    sovits_path = sovits_path
    if show:
        print("推理参数:")
    inp_ref = reference
    how_to_cut = i18n("凑四句一切")
    top_k = 5
    top_p = 1
    temperature = 1
    ref_text_free = False
    speed=1
    if_freeze=False
    inp_refs=[]
    # inp_refs = [
    #     {
    #         "name": "a.wav",
    #         "type": "audio/wav",
    #         "size": 102400,  # 假设大小为102400字节
    #         "data": < file object
    # for a.wav>
    # },
    # {
    # "name": "b.wav",
    # "type": "audio/wav",
    # "size": 204800,  # 假设大小为204800字节
    # "data": < file
    # object
    # for b.wav>
    # },
    # {
    # "name": "c.wav",
    # "type": "audio/wav",
    # "size": 307200,  # 假设大小为307200字节
    # "data": < file
    # object
    # for c.wav>
    # }
    # ]
    if show:
        print(f"inp_ref={inp_ref}\n"
              f"prompt_text={prompt_text}\n"
              f"prompt_language={prompt_language}\n"
              f"text={text}\n"
              f"text_language={text_language}\n"
              f"how_to_cut={how_to_cut}\n"
              f"top_k={top_k}\n"
              f"top_p={top_p}\n"
              f"temperature={temperature}\n"
              f"ref_text_free={ref_text_free}\n"
              f"speed={speed}\n"
              f"if_freeze={if_freeze}\n"
              f"inp_refs={inp_refs}\n")

    change_sovits_weights(sovits_path)
    change_gpt_weights(gpt_path)

    get_tts_wav_generator = get_tts_wav(inp_ref, prompt_text, prompt_language, text, text_language, how_to_cut, top_k,
                                        top_p, temperature, ref_text_free, speed, if_freeze,inp_refs)
    for audio in get_tts_wav_generator:
        output = audio
    if show:
        print(f"output={output}")
    from scipy.io.wavfile import write

    sample_rate, audio_data = output

    audio_data = np.array(audio_data)
    if show:
        print(f"sample_rate,audio_data={sample_rate},{audio_data}")
    output_directory = os.path.join("output", exp_name, reference_name)
    output_filename = output_directory + f"/{text[:10]}"+".wav"
    os.makedirs(output_directory, exist_ok=True)
    write(output_filename, sample_rate, audio_data)
    txt_output_directory = os.path.join(output_directory,f"{text[:10]}.txt")
    with open(txt_output_directory,"w", encoding="utf-8") as file:
        # 写入文本内容
        file.write(f"{text}")
    source_file=inp_ref
    reference_output_directory = os.path.join(output_directory, f"{reference_name}")
    copy_file=os.path.join(output_directory, f"{reference_name}.wav")
    if not os.path.exists(copy_file):
        shutil.copy(source_file, reference_output_directory)
    if show:
        print("输出完成")

# train_file_name="shoulinrui.m4a"
# train=True
# gpu="0-0"
# reference="resources/slice/shoulinrui.m4a_0000063040_0000325440.wav"
# text="兄弟们，现在是凌晨两点半，我搞了整整3个小时，把G最新版本的一键流程跑通了。这次改变在代码上最突出的是，一些预训练模型和权重换了文件名称，支持更多语言代表着语言的字典进行了改变。现在语音识别对应一套字典，文本对应一套字典，我明天把他改一下。其次就是，对内存要求更高。遇到内存爆了，显存炸了很正常，需要自己调节批次大小。然后我写的函数把训练和推理分成了两个，大大降低了作者藏参数的影响。"
# text_language="中文"
# prompt_text="啊，一些重要的公司会以各种主题反复出现在商业就是这样的节目中，例如，啊，星巴克"
# prompt_language="中文"
# autorun2(train_file_name,train,gpu,reference,text,text_language,prompt_text,prompt_language)

# dict_language_v2 = {
#     i18n("中文"): "all_zh",  # 全部按中文识别
#     i18n("英文"): "en",  # 全部按英文识别#######不变
#     i18n("日文"): "all_ja",  # 全部按日文识别
#     i18n("粤语"): "all_yue",  # 全部按中文识别
#     i18n("韩文"): "all_ko",  # 全部按韩文识别
#     i18n("中英混合"): "zh",  # 按中英混合识别####不变
#     i18n("日英混合"): "ja",  # 按日英混合识别####不变
#     i18n("粤英混合"): "yue",  # 按粤英混合识别####不变
#     i18n("韩英混合"): "ko",  # 按韩英混合识别####不变
#     i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
#     i18n("多语种混合(粤语)"): "auto_yue",  # 多语种启动切分识别语种