"""exp096 single-source PM controls. Durations are seconds from kernel entry. MIT-0."""
import os

NVARC_STOP = 3 * 3600
NVARC_TTT = True
QWEN_CONC = 16


def nvarc_settings(env=None):
    env = os.environ if env is None else env
    stop = float(env.get('NVARC_STOP', NVARC_STOP))
    if not 0 < stop <= 5 * 3600:
        raise ValueError('NVARC_STOP must be seconds in (0, 18000]')
    ttt = str(env.get('NVARC_TTT', 'on' if NVARC_TTT else 'off')).lower()
    if ttt not in ('on', 'off'):
        raise ValueError('NVARC_TTT must be on/off')
    return dict(stop=stop, ttt=ttt == 'on')


def train_if_enabled(model, puzzle_ds, formatter, max_seq_length, collator, train_args,
                     fast_model, trainer_type, arguments_type, dataset_type):
    """Keep exp095 training unchanged when enabled; off never constructs a trainer."""
    if not nvarc_settings()['ttt']:
        return model, dict(ttt=False)
    import io
    from contextlib import redirect_stdout, redirect_stderr
    model = fast_model.for_training(model)
    train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
    train_ds = train_ds.cut_to_len(formatter=formatter, name='text', max_len=max_seq_length)
    with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
        trainer = trainer_type(model=model, tokenizer=formatter.tokenizer,
            data_collator=collator, train_dataset=dataset_type.from_list(train_ds.as_list(formatter)),
            dataset_text_field='text', max_seq_length=max_seq_length,
            args=arguments_type(**train_args))
        stats = trainer.train()
        model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
        del trainer
    return model, stats
