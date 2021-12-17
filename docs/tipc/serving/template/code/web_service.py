# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddle_serving_server.web_service import WebService, Op


class TIPCExampleOp(Op):
    """
    ExampleOp for serving server, you can rename by yourself
    """

    def init_op(self):
        """
        initialize the class
        """
        pass

    def preprocess(self, input_dicts, data_id, log_id):
        # preprocess for the inputs
        pass

    def postprocess(self, input_dicts, fetch_dict, data_id, log_id):
        # postprocess for the service output
        pass


class TIPCExampleService(WebService):
    def get_pipeline_response(self, read_op):
        tipc_example_op = TIPCExampleOp(
            name="tipc_example", input_ops=[read_op])
        return tipc_example_op


# define the service class
uci_service = TIPCExampleService(name="tipc_example")
# load config and prepare the service
uci_service.prepare_pipeline_config("config.yml")
# start the service
uci_service.run_service()
