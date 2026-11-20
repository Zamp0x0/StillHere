"""Synthetic world banks for D4 (all values invented for this dataset; no real person, address or number is referenced on purpose).

Every bank is split into four disjoint pools by a keyed hash of the value, so an entity used in TRAIN can never appear in DEV,
SAFETY or HOLDOUT (entity isolation). Composite names (hotels, schools, companies, people) are built from syllable lists so the
pools are large without repeating a template value.
"""
from __future__ import annotations

import hashlib
import itertools
from typing import Dict, List

POOLS = ("TRAIN", "DEV", "SAFETY", "HOLDOUT")
_POOL_SHARES = (("TRAIN", 0.58), ("DEV", 0.14), ("SAFETY", 0.14), ("HOLDOUT", 0.14))
_SALT = "t2d4-entity-pool-v1"


def pool_of(value: str) -> str:
    h = int(hashlib.sha256(f"{_SALT}:{value}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, share in _POOL_SHARES:
        acc += share
        if h < acc:
            return name
    return "HOLDOUT"


CITIES = ["深圳", "广州", "武汉", "成都", "杭州", "南京", "西安", "长沙", "重庆", "苏州", "厦门", "青岛", "大连", "沈阳", "哈尔滨", "长春",
          "昆明", "贵阳", "南宁", "海口", "福州", "合肥", "南昌", "郑州", "济南", "太原", "石家庄", "呼和浩特", "兰州", "银川", "西宁", "乌鲁木齐",
          "拉萨", "天津", "宁波", "无锡", "常州", "温州", "绍兴", "嘉兴", "珠海", "佛山", "东莞", "惠州", "中山", "汕头", "泉州", "漳州",
          "桂林", "柳州", "三亚", "洛阳", "开封", "烟台", "威海", "潍坊", "扬州", "镇江", "徐州", "芜湖", "宜昌", "襄阳", "岳阳", "株洲",
          "绵阳", "乐山", "大理", "丽江", "遵义", "赣州", "九江", "包头", "鞍山", "吉林", "秦皇岛", "保定", "邯郸", "唐山", "湖州", "金华"]

_HOTEL_A = ["云栖", "青禾", "松间", "晴川", "望湖", "半山", "听涛", "星河", "暖阳", "竹里", "枫林", "白鹭", "清溪", "朗月", "知味", "拾光",
            "悦来", "安和", "锦程", "南风", "北辰", "山水", "如归", "嘉禾"]
_HOTEL_B = ["酒店", "宾馆", "客栈", "公寓酒店", "精品酒店", "民宿"]
HOTELS = [a + b for a, b in itertools.product(_HOTEL_A, _HOTEL_B)]

_SURNAMES = ["林", "周", "陈", "王", "李", "张", "刘", "赵", "孙", "吴", "郑", "冯", "何", "许", "宋", "唐", "韩", "曹", "邓", "彭",
             "曾", "萧", "田", "董", "潘", "袁", "蔡", "蒋", "余", "杜", "叶", "程", "魏", "苏", "吕", "丁", "沈", "任", "姚", "卢",
             "钟", "姜", "崔", "谭", "陆", "范", "汪", "廖", "石", "金", "韦", "贾", "夏", "付", "方", "邹", "熊", "白", "孟", "秦"]
_GIVEN = ["峰", "婷", "磊", "静", "涛", "敏", "杰", "琳", "斌", "雪", "超", "燕", "勇", "倩", "鹏", "颖", "强", "晨", "浩", "悦"]
PEOPLE = ["小" + s for s in _SURNAMES] + ["老" + s for s in _SURNAMES] + ["阿" + g for g in _GIVEN] + [s + g for s in _SURNAMES[:30] for g in _GIVEN[:6]]

_SCHOOL_A = ["实验", "外国语", "第一", "第二", "第三", "第五", "第七", "育才", "启明", "城南", "城北", "滨江", "东湖", "西山", "文华", "崇文"]
_SCHOOL_B = ["中学", "小学", "高中", "初中", "职业学院"]
SCHOOLS = [a + b for a, b in itertools.product(_SCHOOL_A, _SCHOOL_B)]

_CO_A = ["蓝桥", "星火", "海川", "远帆", "云图", "极光", "青松", "原点", "知行", "飞鱼", "百川", "长风", "晨曦", "千帆", "灵犀", "砺石"]
_CO_B = ["科技", "设计", "物流", "传媒", "医疗", "教育", "建筑", "食品"]
COMPANIES = [a + b for a, b in itertools.product(_CO_A, _CO_B)]

OCCUPATIONS = ["语文老师", "数学老师", "英语老师", "护士", "程序员", "会计", "设计师", "厨师", "律师", "医生", "工程师", "销售", "摄影师",
               "导游", "司机", "编辑", "药剂师", "建筑师", "翻译", "前台", "健身教练", "幼师", "快递员", "店长", "客服", "银行柜员", "社工",
               "测试工程师", "产品经理", "插画师", "电工", "兽医", "研究员", "公务员", "物业管理员", "保险顾问", "面包师", "园艺师"]
SUBJECTS_TAUGHT = ["语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治", "音乐", "美术", "体育", "信息技术"]
HOBBIES = ["羽毛球", "钓鱼", "爬山", "游泳", "下棋", "弹吉他", "跑步", "摄影", "做饭", "养花", "骑车", "打篮球", "乒乓球", "画画", "唱歌",
           "看电影", "玩桌游", "练书法", "瑜伽", "滑板", "露营", "织毛衣", "打太极", "拼乐高", "攀岩", "打网球", "写字", "听戏", "跳舞", "徒步"]
PETS_KIND = ["猫", "狗", "兔子", "仓鼠", "鹦鹉", "乌龟", "金鱼"]
PET_NAMES = ["豆豆", "雪球", "花卷", "汤圆", "可乐", "布丁", "年糕", "芝麻", "旺财", "糯米", "饺子", "馒头", "毛毛", "球球", "橘子", "栗子",
             "奶茶", "土豆", "小白", "大黄", "团子", "麦兜", "芒果", "椰子", "黑豆", "咖啡", "米粒", "棉花", "可可", "花生", "虎子", "点点"]
FOODS = ["火锅", "米粉", "饺子", "烧烤", "牛肉面", "煎饼", "寿司", "麻辣烫", "炒饭", "包子", "汉堡", "披萨", "螺蛳粉", "烤鸭", "凉皮",
         "小龙虾", "肠粉", "馄饨", "酸菜鱼", "黄焖鸡", "生煎", "冒菜", "热干面", "砂锅粥"]
ITEMS = ["耳机", "自行车", "键盘", "电饭煲", "台灯", "行李箱", "运动鞋", "吹风机", "显示器", "空气炸锅", "电动牙刷", "保温杯", "书包",
         "羽绒服", "平板", "相机", "雨伞", "床垫", "椅子", "手表", "音箱", "滑板", "帐篷", "咖啡机"]
_BRAND_A = ["拓", "岚", "曜", "澜", "芯", "驰", "跃", "朗"]
_BRAND_B = ["米", "达", "途", "格", "瑞", "锐", "克", "诺"]
CAR_MODELS = [a + b + m for a, b in itertools.product(_BRAND_A, _BRAND_B) for m in ("S3", "X5", "Pro")][:150]
CONCERTS = ["音乐节", "话剧", "演唱会", "相声专场", "脱口秀", "livehouse 演出", "交响音乐会", "球赛", "画展", "展会"]
VENUES = [a + b for a, b in itertools.product(["东方", "滨江", "城西", "未来", "体育", "文化", "星光", "湖畔"], ["剧院", "体育馆", "艺术中心", "会展中心", "大剧场"])]
TRANSPORT = ["地铁", "公交", "骑车", "开车", "走路", "打车", "坐班车", "电动车"]
CLINICS = [a + b for a, b in itertools.product(["仁和", "康宁", "和美", "安康", "济民", "博爱"], ["诊所", "口腔诊所", "门诊部", "医院"])]
MAJORS = ["计算机", "会计", "护理", "法学", "机械", "新闻", "汉语言", "金融", "建筑", "日语", "心理学", "生物", "统计", "环境工程"]
RELATIONS = {"friend": ["朋友", "好朋友", "老同学"], "spouse": ["老婆", "老公", "对象"], "family": ["表弟", "表姐", "堂哥", "妹妹", "弟弟", "姐姐", "我妈", "我爸"],
             "colleague": ["同事", "领导", "室友"], "neighbor": ["邻居"]}
WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
CLOCK = ["七点", "七点半", "八点", "八点半", "九点", "九点半", "十点", "十点半", "两点", "两点半", "三点", "四点", "五点半", "六点"]
DISEASES = ["胃病", "过敏性鼻炎", "腰椎问题", "高血压", "失眠", "偏头痛"]


def pooled(bank: List[str]) -> Dict[str, List[str]]:
    out = {p: [] for p in POOLS}
    for v in dict.fromkeys(bank):
        out[pool_of(v)].append(v)
    return out


BANKS = {name: pooled(v) for name, v in {
    "city": CITIES, "hotel": HOTELS, "person": PEOPLE, "school": SCHOOLS, "company": COMPANIES, "occupation": OCCUPATIONS,
    "subject_taught": SUBJECTS_TAUGHT, "hobby": HOBBIES, "pet_name": PET_NAMES, "food": FOODS, "item": ITEMS, "car": CAR_MODELS,
    "concert": CONCERTS, "venue": VENUES, "transport": TRANSPORT, "clinic": CLINICS, "major": MAJORS, "disease": DISEASES,
}.items()}

# Banks that are closed vocabularies (small, shared by every pool): isolating them would leave a pool empty. They are never the only
# isolated element of a scenario: persona, people, places and numbers still differ across pools.
SHARED_BANKS = ("pet_kind", "weekday", "clock", "transport", "concert", "subject_taught", "disease")
BANKS.update({"pet_kind": {p: PETS_KIND for p in POOLS}, "weekday": {p: WEEKDAYS for p in POOLS}, "clock": {p: CLOCK for p in POOLS},
              "transport": {p: TRANSPORT for p in POOLS}, "concert": {p: CONCERTS for p in POOLS},
              "subject_taught": {p: SUBJECTS_TAUGHT for p in POOLS}, "disease": {p: DISEASES for p in POOLS}})


def all_entity_values() -> set:
    """Every named value any pool can produce (for the grounding validator's entity scan)."""
    out = set()
    for name, pools in BANKS.items():
        for vs in pools.values():
            out.update(vs)
    return out
