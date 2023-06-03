import sys
import transformers4rec as t4r
from transformers4rec.torch.utils.examples_utils import fit_and_evaluate
import numpy as np
from merlin.core.utils import download_file
from merlin.io import *
from merlin_standard_lib import Schema
from transformers4rec import torch as tr

start_time_index = 3
end_time_index = 284


def set_value_count_max_value(schema, max_sequence_length):
    for i in schema:
        i.value_count.max = max_sequence_length


def delete_some_features(schema,
                         genre=False,
                         genome=False,
                         tag=False,
                         genome_relevance=False,
                         rating=False,
                         day=False,
                         week=False,
                         year=False,
                         sin=False):
    if not genre:
        schema = schema.remove_by_name("_genres-list_seq")
    if not genome:
        schema = schema.remove_by_name("genome_tag-list_seq")
    if not tag:
        schema = schema.remove_by_name("tag-list_seq")
    if not genome_relevance:
        schema = schema.remove_by_name("genome_relevance-list_seq")
    if not rating:
        schema = schema.remove_by_name("rating-list_seq")
    if not day:
        schema = schema.remove_by_name("et_dayofday-list_seq")
    if not week:
        schema = schema.remove_by_name("et_dayofweek-list_seq")
    if not year:
        schema = schema.remove_by_name("et_year-list_seq")
    if not sin:
        schema = schema.remove_by_name("et_dayofweek_sin-list_seq")
    return schema


L = [5]


def run():
    for max_sequence_length in L:
        for cont in ["rating", "genome_relevance"]:

            config = dict()
            config[cont] = True

            SCHEMA_PATH = f"model_configs_and_results/cont/{max_sequence_length}/schema_demo.pb"

            sys.stdout = open(f'model_configs_and_results/cont/{max_sequence_length}/print_{cont}.txt', 'wt')

            print("max_sequence_length:", max_sequence_length)

            schema = Schema().from_proto_text(SCHEMA_PATH)

            schema = delete_some_features(schema,
                                          **config)

            set_value_count_max_value(schema, max_sequence_length)

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

            transformer_config = tr.XLNetConfig.build(
                d_model=d_model, n_head=8, n_layer=2, total_seq_length=max_sequence_length
            )

            model = transformer_config.to_torch_model(input_module, prediction_task)

            training_args = tr.trainer.T4RecTrainingArguments(
                output_dir=f"temp/tmp{max_sequence_length}_withF",
                max_sequence_length=max_sequence_length,
                data_loader_engine='nvtabular',
                num_train_epochs=10,
                dataloader_drop_last=False,
                per_device_train_batch_size=384,
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

            OT_results = fit_and_evaluate(recsys_trainer,
                                          start_time_index=start_time_index, end_time_index=end_time_index,
                                          input_dir=f'./datasets/preproc_sessions_by_day{max_sequence_length}'
                                          )

            avg_results = {k: np.mean(v) for k, v in OT_results.items()}
            for key in sorted(avg_results.keys()):
                print(" %s = %s" % (key, str(avg_results[key])))


run()
