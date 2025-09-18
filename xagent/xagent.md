

机器人硬件：
一个xarm机械臂，带有一个hand-camera
机器人软件:
2D感知API：
	dinox-detect api, （2D检测）
	dinox-refer api,   (2d指代检测)
	dinox-grasp api,(检测抓取点)
决策大模型LLM或VLM：模型选型待定
	llm或vlm api
机器人操作控制API：
	pick_n_place的API
    其他常规控制API

目标：利用机器人硬件和软件，展现高级语义理解抓取能力（reorder）
从左到右依次排列红，黄，蓝三种颜色袋子，通过语音指令可以交换袋子顺序，比如“将左边的红色袋子挪到中间”，机械臂识别到中间袋子是黄色不是红色，将黄色袋子和红色袋子进行交换。


任务：设计一个agents解决方案