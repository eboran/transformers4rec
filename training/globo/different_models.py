import sys
import transformers4rec as t4r
from transformers4rec.torch.utils.examples_utils import fit_and_evaluate
import numpy as np
from merlin.core.utils import download_file
from merlin.io import *
from merlin_standard_lib import Schema
from transformers4rec import torch as tr

start_time_index = 24
end_time_index = 72

max_sequence_length = 20

models = [
    "AlbertConfig",
    "XLNetConfig",
    "ElectraConfig"
]


def get_config(d_model, max_sequence_length, model_name):
    print("d_model, max_sequence_length, model_name", d_model, max_sequence_length, model_name)

    if model_name == "AlbertConfig":
        return tr.AlbertConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    if model_name == "ElectraConfig":
        return tr.ElectraConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    if model_name == "GPT2Config":
        return tr.GPT2Config.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    if model_name == "LongformerConfig":
        return tr.LongformerConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    if model_name == "ReformerConfig":
        return tr.ReformerConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    if model_name == "TransfoXLConfig":
        return tr.TransfoXLConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    if model_name == "XLNetConfig":
        return tr.XLNetConfig.build(
            d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
        )
    raise f"error in {model_name}"


def run():
    for model_name in models:

        SCHEMA_PATH = f"model_configs_and_results/different-models/{max_sequence_length}/schema_demo.pb"

        sys.stdout = open(f'model_configs_and_results/different-models/{max_sequence_length}/{model_name}.txt', 'wt')

        print("max_sequence_length:", max_sequence_length)

        schema = Schema().from_proto_text(SCHEMA_PATH)

        print(schema)

        for s in schema:
            print(s, s.value_count)

        d_model = 64
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=64,
            aggregation="concat",
            d_output=d_model,
            masking="mlm",
        )

        prediction_task = tr.NextItemPredictionTask(hf_format=True, weight_tying=True)

        transformer_config = get_config(d_model, max_sequence_length, model_name)

        model = transformer_config.to_torch_model(input_module, prediction_task)

        training_args = tr.trainer.T4RecTrainingArguments(
            output_dir=f"temp/tmp{max_sequence_length}_withF",
            max_sequence_length=max_sequence_length,
            data_loader_engine='nvtabular',
            num_train_epochs=10,
            dataloader_drop_last=False,
            per_device_train_batch_size=320,
            per_device_eval_batch_size=512,
            learning_rate=0.0001,
            fp16=True,
            report_to=[],
            logging_steps=10
        )

        recsys_trainer = tr.Trainer(
            model=model,
            args=training_args,
            schema=schema,
            compute_metrics=True)

        schema: tr.Schema = tr.data.tabular_sequence_testing_data.schema

        d_model = 64

        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
            continuous_projection=d_model,
            aggregation="concat",
            masking="causal",
        )
        prediction_tasks = tr.NextItemPredictionTask()

        transformer_config = tr.XLNetConfig.build(
            d_model=d_model, n_head=4, n_layer=2, total_seq_length=max_sequence_length
        )
        model: tr.Model = transformer_config.to_torch_model(input_module, prediction_tasks)

        input_dir = f'./datasets/preproc_sessions_renamed'

        OT_results = fit_and_evaluate(recsys_trainer,
                                      start_time_index=start_time_index, end_time_index=end_time_index,
                                      input_dir=input_dir
                                      )

        avg_results = {k: np.mean(v) for k, v in OT_results.items()}
        for key in sorted(avg_results.keys()):
            print(" %s = %s" % (key, str(avg_results[key])))

        import pickle

        with open(f'avg_results_{model_name}.pickle', 'wb') as handle:
            pickle.dump(avg_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


run()
