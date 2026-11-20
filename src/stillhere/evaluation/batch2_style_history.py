"""Synthetic STYLE history for the Batch 2 memory personas (written for Batch 2; public, no real person).

The memory worlds' own messages carry the facts the questions ask about, so they are NOT used as style examples (an example that states the
answer would let a reply "preserve" a fact by copying the prompt's example instead of using memory). These lines are small talk with no
fact of the worlds (checked by hand against each world's facts: no job, place, hobby, person or health detail), one register per persona, so the length policy differs between personas (terse / medium / chatty).
"""
from __future__ import annotations

from typing import Dict, List

VERSION = "b2-style-history-3"   # 2: + the Batch 2.1 personas (the earlier seven are unchanged)

HISTORY: Dict[str, List[str]] = {
    # writer_dataset_v3
    "syn.ra": ["行", "可以啊", "我看看", "晚点说", "没问题", "好嘞", "收到", "周末再约", "那就这样", "有空聊", "嗯嗯", "改天吧"],
    "syn.rb": ["哈哈哈好呀～", "今天又是忙碌的一天呢", "这个我觉得挺好的呀", "忙完啦，终于可以歇会儿了", "你也早点休息哦", "嗯嗯明白啦",
               "这周天气好舒服", "我晚点回你哈", "好的好的没问题～", "哈哈哈你太逗了", "我也是这么想的呢", "有空一起吃饭呀"],
    "syn.rc": ["今天早上空气不错，出去走了一圈，回来泡了壶茶，挺舒服的。", "年纪大了，晚上睡得早，早上醒得也早。", "你们年轻人工作忙，注意身体，别老熬夜。",
               "这几天降温了，出门记得多穿点。", "好的，我知道了，你忙你的吧。", "前阵子整理了一下旧书，翻出来不少有意思的。", "慢慢来，不着急。",
               "嗯，有空回来吃饭。", "最近小区里新开了家早点铺，味道还行。", "电话里说不清楚，等见面再细说。", "我这边一切都好，不用惦记。", "行，就这么办吧。"],
    # writer_dataset_v4 (Batch 2.1; checked by hand against the v4 facts: no job, place, food, drink, sport, pet, person or event)
    "syn.ta": ["好呀", "收到～", "等我一下", "晚点聊哈", "可以可以", "哈哈好的", "我看看再说", "没问题呀", "嗯嗯", "刚忙完", "辛苦啦", "明天说吧"],
    "syn.tb": ["行", "好", "知道了", "等会儿", "没事", "可以", "嗯", "回头说", "忙着呢", "收到", "就这样", "好嘞"],
    "syn.tc": ["好的，我知道了，你们忙你们的。", "今天天气挺好，出门走了走，心情也好。", "年纪大了，记性不太好，有事多提醒我。", "你们注意身体，别太累了。",
               "嗯，等有空再慢慢说。", "电话里讲不清，见面再聊吧。", "我这边都好，不用挂念。", "最近睡得还算踏实。", "好，就按你们说的办。",
               "这两天降温了，记得添衣服。", "慢慢来，不着急的。", "有空就回来坐坐。"],
    # writer_dataset_v5 (Batch 2.1 post-GPU retest; checked against the v5 facts: no job, place, school, sport, food, person or event)
    "syn.ua": ["嗯", "行", "知道了", "好", "等下", "在忙", "可以", "没事", "收到", "回头说", "OK", "哦"],
    "syn.ub": ["今天下午开了一个很长的会，讨论了好多细节，回到家已经七点多了，饭都还没顾上吃，不过总算把事情都理清楚了。",
               "你说的这个我觉得挺有道理的，不过我还是想再考虑一下，毕竟这种事情急不得，慢慢来反而更稳妥一些。",
               "最近天气变化特别大，早上出门还挺凉的，中午就热得穿不住外套了，大家一定要注意别感冒了。",
               "昨天晚上看了一部老电影，讲的是一家人之间的故事，看到最后有点感动，好久没有这种感觉了。",
               "其实我一直觉得，很多事情只要认真去做，结果一般都不会太差，就算差一点也能学到不少东西。",
               "周末本来想好好休息一下的，结果家里事情一大堆，忙了两天也没歇着，不过心里倒是挺踏实的。",
               "这个问题我之前也遇到过，当时想了很多办法都没用，后来换了个思路才慢慢解决，所以你别着急。",
               "说起来真是挺巧的，我刚想给你发消息，你就先问我了，看来我们想到一块儿去了，哈哈。",
               "今天早上路上特别堵，我比平时早出门了二十分钟，结果到的时间还是差不多，真是没办法。",
               "我最近在试着每天早点睡，虽然一开始不太习惯，但坚持了几天之后感觉精神好了很多。",
               "你要是有空的话，我们可以找个时间好好聊一聊，电话里有些事情说不太清楚，当面讲更方便。",
               "谢谢你一直这么关心，我这边一切都还好，有什么需要帮忙的我一定会第一时间跟你说的。"],
    "syn.uc": ["收到\n我看看", "刚到\n累死", "行\n就这么办", "等我一下\n马上", "哈哈\n真的假的", "明天吧\n今天太晚了", "对\n没错", "好的\n谢了",
               "在路上\n晚点回", "可以\n没问题", "啥情况\n说说", "睡了\n明天聊"],
    # writer_dataset_v2
    "syn.qa": ["好", "可以", "等下", "ok", "行吧", "我想想", "明天说", "没事", "嗯", "收到了", "有道理", "再说吧"],
    "syn.qb": ["今天下班好晚呀", "嗯嗯好的", "你吃饭了没", "我刚到家呢", "这个可以的", "哈哈哈哈", "累死了今天", "好呀好呀",
               "周末想睡个懒觉", "你也注意身体", "我晚点打给你", "没关系的"],
    "syn.qc": ["刚忙完，等会儿回你", "这个我得查一下", "可以啊，我都行", "最近事情有点多", "哈哈确实", "我觉得还行吧", "等我忙完这阵",
               "行，那就这么定", "我朋友也这么说", "晚上出去走走", "好的谢谢", "改天细聊"],
    "syn.qd": ["好的，知道了。", "最近还行，就是有点忙。", "你们忙就不用总打电话了。", "今天去公园走了走，人挺多的。", "嗯，记住了。",
               "天冷了，你们也多穿点衣服。", "我这边都挺好的，放心吧。", "晚上看了会儿电视就睡了。", "今天又和人下了两盘棋。", "有空回来看看就行。",
               "电话费我交过了。", "行，听你们的。"],
}


def history(persona_id: str) -> List[str]:
    return list(HISTORY[persona_id])
