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


def delete_some_features(schema,
                         click_environment=False,
                         click_deviceGroup=False,
                         click_country=False,
                         click_os=False,
                         click_region=False,
                         click_referrer_type=False,
                         ):
    schema = schema.remove_by_name("item_age_hours_norm")
    schema = schema.remove_by_name("weekday_sin")
    schema = schema.remove_by_name("weekday_cos")
    schema = schema.remove_by_name("hour_sin")
    schema = schema.remove_by_name("hour_cos")

    if not click_environment:
        schema = schema.remove_by_name("click_environment")
    if not click_deviceGroup:
        schema = schema.remove_by_name("click_deviceGroup")
    if not click_country:
        schema = schema.remove_by_name("click_country")
    if not click_os:
        schema = schema.remove_by_name("click_os")
    if not click_region:
        schema = schema.remove_by_name("click_region")
    if not click_referrer_type:
        schema = schema.remove_by_name("click_referrer_type")

    return schema


max_sequence_length = 20


def run():
    for cat in ["click_environment", "click_deviceGroup", "click_country", "click_os", "click_region", "click_referrer_type"]:

        config = dict()
        config[cat] = True

        SCHEMA_PATH = f"model_configs_and_results/cat/{max_sequence_length}/schema_demo.pb"

        sys.stdout = open(f'model_configs_and_results/cat/{max_sequence_length}/print_{cat}.txt', 'wt')

        print("max_sequence_length:", max_sequence_length)

        schema = Schema().from_proto_text(SCHEMA_PATH)

        schema = delete_some_features(schema,
                                      **config)

        for s in schema:
            print(s, s.value_count)

        d_model = 64
        input_module = tr.TabularSequenceFeatures.from_schema(
            schema,
            max_sequence_length=max_sequence_length,
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
                                      input_dir=f'./datasets/preproc_sessions_renamed'
                                      )

        avg_results = {k: np.mean(v) for k, v in OT_results.items()}
        for key in sorted(avg_results.keys()):
            print(" %s = %s" % (key, str(avg_results[key])))


run()
