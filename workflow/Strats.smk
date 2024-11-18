
workdir: "/home/srmish/strats/src"

rule all:
    input:
        expand(
            "results/mimic/{fraction}_{fold}.SUCCESS",
            fraction=[f"{i/10:1.1f}" for i in range(6, 11)],
            fold=range(1, 11)
        ),
        expand(
            "results/grud/{fraction}_{fold}.SUCCESS",
            fraction=[f"{i/10:1.1f}" for i in range(6, 11)],
            fold=range(1, 11)
        )


rule mimic:
    output:
        "results/mimic/{fraction}_{fold}.SUCCESS",
    log:
        "logs/mimic/{fraction}_{fold}.log",
    # conda:
    #     ...
    resources:
        mem_mb=16*1024,
        gpu=1,
        runtime='20h',
    shell:
        "python main.py --dataset mimic_iii --model_type strats --hid_dim 64 --num_layers 2 "
        "--num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-5 "
        "--load_ckpt_path ../outputs/mimic_iii_original_strats/pretrain/checkpoint_best.bin "
        "--run {wildcards.fold}o10 --train_frac {wildcards.fraction} --patience 10 "
        "> {log} 2> {log} && touch {output}"



rule grud:
    output:
        "results/grud/{fraction}_{fold}.SUCCESS",
    log:
        "logs/grud/{fraction}_{fold}.log",
    # conda:
    #     ...
    resources:
        mem_mb=16*1024,
        gpu=1,
        runtime='20h',
    shell:
        "python main.py --dataset mimic_iii --model_type grud --hid_dim 64 --dropout 0.2 --lr 5e-4 "
        "--run {wildcards.fold}o10 --train_frac {wildcards.fraction} --patience 10 "
        "> {log} 2> {log} && touch {output}"
