import os
import random
import string
import subprocess
from time import sleep, time

import allure
import psutil
from openai import OpenAI
from pytest_assume.plugin import assume
from utils.config_utils import get_cuda_prefix_by_workerid, get_workerid
from utils.get_run_config import get_command_with_extra
from utils.rule_condition_assert import assert_result
from utils.run_client_chat import command_line_test

from lmdeploy.serve.openai.api_client import APIClient
from lmdeploy.utils import is_bf16_supported

BASE_HTTP_URL = 'http://localhost'
DEFAULT_PORT = 23333


def start_restful_api(config, param, model, model_path, backend_type, worker_id):
    log_path = config.get('log_path')

    cuda_prefix = param['cuda_prefix']
    tp_num = param['tp_num']
    if 'extra' in param.keys():
        extra = param['extra']
    else:
        extra = None

    if 'modelscope' in param.keys():
        modelscope = param['modelscope']
        if modelscope:
            os.environ['LMDEPLOY_USE_MODELSCOPE'] = 'True'
            model_path = model

    if cuda_prefix is None:
        cuda_prefix = get_cuda_prefix_by_workerid(worker_id, tp_num=tp_num)

    worker_num = get_workerid(worker_id)
    if worker_num is None:
        port = DEFAULT_PORT
    else:
        port = DEFAULT_PORT + worker_num

    cmd = get_command_with_extra('lmdeploy serve api_server ' + model_path + ' --session-len 8096 --server-port ' +
                                 str(port),
                                 config,
                                 model,
                                 need_tp=True,
                                 cuda_prefix=cuda_prefix,
                                 extra=extra)

    if backend_type == 'turbomind':
        if ('w4' in model or '4bits' in model or 'awq' in model.lower()):
            cmd += ' --model-format awq'
        elif 'gptq' in model.lower():
            cmd += ' --model-format gptq'
    if backend_type == 'pytorch':
        cmd += ' --backend pytorch'
        if not is_bf16_supported():
            cmd += ' --dtype float16'
    if 'quant_policy' in param.keys() and param['quant_policy'] is not None:
        quant_policy = param['quant_policy']
        cmd += f' --quant-policy {quant_policy}'

    if not is_bf16_supported():
        cmd += ' --cache-max-entry-count 0.5'

    start_log = os.path.join(log_path, 'start_restful_' + model.split('/')[1] + worker_id + '.log')

    print('reproduce command restful: ' + cmd)

    with open(start_log, 'w') as f:
        f.writelines('reproduce command restful: ' + cmd + '\n')

        startRes = subprocess.Popen([cmd], stdout=f, stderr=f, shell=True, text=True, encoding='utf-8')
        pid = startRes.pid

    http_url = BASE_HTTP_URL + ':' + str(port)
    with open(start_log, 'r') as file:
        content = file.read()
        print(content)
    start_time = int(time())

    start_timeout = 300
    if not is_bf16_supported():
        start_timeout = 600

    sleep(5)
    for i in range(start_timeout):
        sleep(1)
        end_time = int(time())
        total_time = end_time - start_time
        result = health_check(http_url)
        if result or total_time >= start_timeout:
            break
    allure.attach.file(start_log, attachment_type=allure.attachment_type.TEXT)
    return pid, startRes


def stop_restful_api(pid, startRes, param):
    if pid > 0:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    if 'modelscope' in param.keys():
        modelscope = param['modelscope']
        if modelscope:
            del os.environ['LMDEPLOY_USE_MODELSCOPE']


def run_all_step(config, cases_info, worker_id: str = '', port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)

    model = get_model(http_url)

    if model is None:
        assert False, 'server not start correctly'
    for case in cases_info.keys():
        if ('coder' in model.lower() or 'codellama' in model.lower()) and 'code' not in case:
            continue

        case_info = cases_info.get(case)

        with allure.step(case + ' step1 - command chat regression'):
            chat_result, chat_log, msg = command_line_test(config, case, case_info, model, 'api_client', http_url,
                                                           worker_id)
            allure.attach.file(chat_log, attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert chat_result, msg

        with allure.step(case + ' step2 - restful_test - openai chat'):
            restful_result, restful_log, msg = open_chat_test(config, case, case_info, model, http_url, worker_id)
            allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)
        with assume:
            assert restful_result, msg

        with allure.step(case + ' step3 - restful_test - interactive chat'):
            active_result, interactive_log, msg = interactive_test(config, case, case_info, model, http_url, worker_id)
            allure.attach.file(interactive_log, attachment_type=allure.attachment_type.TEXT)

        with assume:
            assert active_result, msg


def open_chat_test(config, case, case_info, model, url, worker_id: str = ''):
    log_path = config.get('log_path')

    restful_log = os.path.join(log_path, 'restful_' + model + worker_id + '_' + case + '.log')

    file = open(restful_log, 'w')

    result = True

    api_client = APIClient(url)
    model_name = api_client.available_models[0]

    messages = []
    msg = ''
    for prompt_detail in case_info:
        if not result:
            break
        prompt = list(prompt_detail.keys())[0]
        messages.append({'role': 'user', 'content': prompt})
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_completions_v1(model=model_name, messages=messages, top_k=1, max_tokens=256):
            output_message = output.get('choices')[0].get('message')
            messages.append(output_message)

            output_content = output_message.get('content')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content, prompt_detail.values(), model_name)
            file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
            if not case_result:
                msg += reason
            result = result & case_result
    file.close()
    return result, restful_log, msg


def interactive_test(config, case, case_info, model, url, worker_id: str = ''):
    log_path = config.get('log_path')

    interactive_log = os.path.join(log_path, 'interactive_' + model + worker_id + '_' + case + '.log')

    file = open(interactive_log, 'w')

    result = True

    api_client = APIClient(url)
    file.writelines('available_models:' + ','.join(api_client.available_models) + '\n')

    # Randomly generate 6 characters and concatenate them into a string.
    characters = string.digits
    random_chars = ''.join(random.choice(characters) for i in range(6))

    messages = []
    msg = ''
    for prompt_detail in case_info:
        prompt = list(prompt_detail.keys())[0]
        new_prompt = {'role': 'user', 'content': prompt}
        messages.append(new_prompt)
        file.writelines('prompt:' + prompt + '\n')

        for output in api_client.chat_interactive_v1(prompt=prompt,
                                                     interactive_mode=True,
                                                     session_id=random_chars,
                                                     top_k=1,
                                                     request_output_len=256):
            output_content = output.get('text')
            file.writelines('output:' + output_content + '\n')

            case_result, reason = assert_result(output_content, prompt_detail.values(), model)
            file.writelines('result:' + str(case_result) + ',reason:' + reason + '\n')
            if not case_result:
                msg += reason
            result = result & case_result
    file.close()
    return result, interactive_log, msg


def health_check(url):
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        messages = []
        messages.append({'role': 'user', 'content': '你好'})
        for output in api_client.chat_completions_v1(model=model_name, messages=messages, top_k=1):
            if output.get('code') is not None and output.get('code') != 0:
                return False
            return True
    except Exception:
        return False


def get_model(url):
    try:
        api_client = APIClient(url)
        model_name = api_client.available_models[0]
        return model_name.split('/')[-1]
    except Exception:
        return None


PIC = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg'  # noqa E501
PIC2 = 'https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/demo/resources/human-pose.jpg'  # noqa E501


def run_vl_testcase(config, port: int = DEFAULT_PORT):
    http_url = BASE_HTTP_URL + ':' + str(port)
    log_path = config.get('log_path')

    client = OpenAI(api_key='YOUR_API_KEY', base_url=http_url + '/v1')
    model_name = client.models.list().data[0].id

    restful_log = os.path.join(log_path, 'restful_vl_' + model_name.split('/')[-1] + str(port) + '.log')
    file = open(restful_log, 'w')

    prompt_messages = [{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC,
            },
        }, {
            'type': 'image_url',
            'image_url': {
                'url': PIC2,
            },
        }],
    }]

    response = client.chat.completions.create(model=model_name, messages=prompt_messages, temperature=0.8, top_p=0.8)
    file.writelines(str(response).lower() + '\n')

    api_client = APIClient(http_url)
    model_name = api_client.available_models[0]
    for item in api_client.chat_completions_v1(model=model_name, messages=prompt_messages):
        continue
    file.writelines(str(item) + '\n')
    file.close()

    allure.attach.file(restful_log, attachment_type=allure.attachment_type.TEXT)

    assert 'tiger' in str(response).lower() or '虎' in str(response).lower() or 'ski' in str(
        response).lower() or '滑雪' in str(response).lower(), response
    assert 'tiger' in str(item).lower() or '虎' in str(item).lower() or 'ski' in str(item).lower() or '滑雪' in str(
        item).lower(), item
