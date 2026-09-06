"""
Data Generator for Structured IE Evaluation.
Creates:
1. synthetic_contrast_pairs.jsonl: 24 pairs (48 items)
2. cross_domain_dev.jsonl: 24 items
3. cross_domain_test.jsonl: 48 items
4. robustness_edge_cases.jsonl: 16 items
5. novel_dev_24.jsonl: 24 items (including the 6 corrected smoke windows)
6. novel_test_96.jsonl: 96 items
7. long_text_fixtures.jsonl: 12 items
"""

import os
import json

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


def build_synthetic_contrast_pairs():
    pairs = []

    # 1. Direction Reversal Pairs (6 pairs = 12 items)
    # Pair 1: Mentor-Apprentice
    pairs.append({
        "pair_id": "dir_01",
        "variant": "A",
        "text": "玄空子大师在丹塔大殿中当众宣布，正式收青年药师萧炎为关门弟子。",
        "gold_entities": [
            {"text": "玄空子", "type": "Person", "span": [0, 3]},
            {"text": "丹塔", "type": "Organization", "span": [7, 9]},
            {"text": "萧炎", "type": "Person", "span": [21, 23]}
        ],
        "gold_relations": [
            {"type": "mentor_of", "subject": "玄空子", "object": "萧炎", "polarity": "positive", "modality": "actual"}
        ]
    })
    pairs.append({
        "pair_id": "dir_01",
        "variant": "B",
        "text": "青年药师萧炎在丹塔大殿中当众宣布，正式拜玄空子大师为授业恩师。",
        "gold_entities": [
            {"text": "萧炎", "type": "Person", "span": [4, 6]},
            {"text": "丹塔", "type": "Organization", "span": [7, 9]},
            {"text": "玄空子", "type": "Person", "span": [19, 22]}
        ],
        "gold_relations": [
            {"type": "mentor_of", "subject": "玄空子", "object": "萧炎", "polarity": "positive", "modality": "actual"}
        ]
    })

    # Pair 2: Parent-Child
    pairs.append({
        "pair_id": "dir_02",
        "variant": "A",
        "text": "萧战注视着眼前的少年，他作为萧炎的亲生父亲，眼中满是欣慰之色。",
        "gold_entities": [
            {"text": "萧战", "type": "Person", "span": [0, 2]},
            {"text": "萧炎", "type": "Person", "span": [15, 17]}
        ],
        "gold_relations": [
            {"type": "parent_of", "subject": "萧战", "object": "萧炎", "polarity": "positive", "modality": "actual"}
        ]
    })
    pairs.append({
        "pair_id": "dir_02",
        "variant": "B",
        "text": "少年萧炎快步走上前去，恭敬地向自己的亲生父亲萧战躬身行礼问安。",
        "gold_entities": [
            {"text": "萧炎", "type": "Person", "span": [2, 4]},
            {"text": "萧战", "type": "Person", "span": [20, 22]}
        ],
        "gold_relations": [
            {"type": "parent_of", "subject": "萧战", "object": "萧炎", "polarity": "positive", "modality": "actual"}
        ]
    })

    # Pair 3: Corporate Shareholding
    pairs.append({
        "pair_id": "dir_03",
        "variant": "A",
        "text": "天风投资集团通过定向增发，正式成为华云科技公司的第一大控股股东。",
        "gold_entities": [
            {"text": "天风投资集团", "type": "Organization", "span": [0, 6]},
            {"text": "华云科技公司", "type": "Organization", "span": [15, 21]}
        ],
        "gold_relations": [
            {"type": "holds_equity", "subject": "天风投资集团", "object": "华云科技公司", "polarity": "positive", "modality": "actual"}
        ]
    })
    pairs.append({
        "pair_id": "dir_03",
        "variant": "B",
        "text": "华云科技公司通过反向收购，反向控股了天风投资集团百分之五十一的股权。",
        "gold_entities": [
            {"text": "华云科技公司", "type": "Organization", "span": [0, 6]},
            {"text": "天风投资集团", "type": "Organization", "span": [17, 23]}
        ],
        "gold_relations": [
            {"type": "holds_equity", "subject": "华云科技公司", "object": "天风投资集团", "polarity": "positive", "modality": "actual"}
        ]
    })

    # Pair 4: Corporate Employment / Tenure
    pairs.append({
        "pair_id": "dir_04",
        "variant": "A",
        "text": "董事会全票表决通过，任命李明远先生出任盛元重工集团首席执行官。",
        "gold_entities": [
            {"text": "李明远", "type": "Person", "span": [12, 15]},
            {"text": "盛元重工集团", "type": "Organization", "span": [19, 25]}
        ],
        "gold_relations": [
            {"type": "holds_position", "subject": "李明远", "object": "盛元重工集团", "role": "首席执行官", "polarity": "positive", "modality": "actual"}
        ]
    })
    pairs.append({
        "pair_id": "dir_04",
        "variant": "B",
        "text": "盛元重工集团在年报中确认，李明远先生已被免除该集团首席执行官职务。",
        "gold_entities": [
            {"text": "盛元重工集团", "type": "Organization", "span": [0, 6]},
            {"text": "李明远", "type": "Person", "span": [13, 16]}
        ],
        "gold_relations": [
            {"type": "holds_position", "subject": "李明远", "object": "盛元重工集团", "role": "首席执行官", "polarity": "negative", "modality": "actual"}
        ]
    })

    # Pair 5: Transaction Transfer
    pairs.append({
        "pair_id": "dir_05",
        "variant": "A",
        "text": "根据结算单据，供货商宏达商贸已将八十万元采购退款原路退还至买方天宇实业。",
        "gold_entities": [
            {"text": "宏达商贸", "type": "Organization", "span": [10, 14]},
            {"text": "八十万元", "type": "Amount", "span": [17, 21]},
            {"text": "天宇实业", "type": "Organization", "span": [32, 36]}
        ],
        "gold_relations": [
            {"type": "transferred_to", "subject": "宏达商贸", "object": "天宇实业", "polarity": "positive", "modality": "actual"}
        ]
    })
    pairs.append({
        "pair_id": "dir_05",
        "variant": "B",
        "text": "根据结算单据，买方天宇实业已将八十万元采购定金全额电汇至供货商宏达商贸。",
        "gold_entities": [
            {"text": "天宇实业", "type": "Organization", "span": [9, 13]},
            {"text": "八十万元", "type": "Amount", "span": [16, 20]},
            {"text": "宏达商贸", "type": "Organization", "span": [31, 35]}
        ],
        "gold_relations": [
            {"type": "transferred_to", "subject": "天宇实业", "object": "宏达商贸", "polarity": "positive", "modality": "actual"}
        ]
    })

    # Pair 6: Clan / Sect Membership
    pairs.append({
        "pair_id": "dir_06",
        "variant": "A",
        "text": "白袍老者云棱拱手向各方宣布，纳兰嫣然乃是云岚宗现任少宗主。",
        "gold_entities": [
            {"text": "云棱", "type": "Person", "span": [4, 6]},
            {"text": "纳兰嫣然", "type": "Person", "span": [14, 18]},
            {"text": "云岚宗", "type": "Organization", "span": [21, 24]}
        ],
        "gold_relations": [
            {"type": "member_of", "subject": "纳兰嫣然", "object": "云岚宗", "role": "少宗主", "polarity": "positive", "modality": "actual"}
        ]
    })
    pairs.append({
        "pair_id": "dir_06",
        "variant": "B",
        "text": "纳兰嫣然在家族议事厅明确声明，纳兰肃长老并不代表云岚宗立场。",
        "gold_entities": [
            {"text": "纳兰嫣然", "type": "Person", "span": [0, 4]},
            {"text": "纳兰肃", "type": "Person", "span": [15, 18]},
            {"text": "云岚宗", "type": "Organization", "span": [25, 28]}
        ],
        "gold_relations": []  # No membership relation supported!
    })

    # 2. Modality / Negation / Condition Pairs (6 pairs = 12 items)
    for i in range(1, 7):
        p_id = f"mod_{i:02d}"
        if i == 1:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "经双方财务主管现场核对无误，启明智造已将五百万元设备尾款全额转入立新精机账户。",
                "gold_events": [{"event_type": "Payment", "roles": {"payer": "启明智造", "payee": "立新精机", "amount": "五百万元"}, "modality": "actual", "polarity": "positive"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "根据初步合作意向书草案，启明智造拟在验收达标后向立新精机支付五百万元设备尾款。",
                "gold_events": [{"event_type": "Payment", "roles": {"payer": "启明智造", "payee": "立新精机", "amount": "五百万元"}, "modality": "intended", "polarity": "positive"}]
            })
        elif i == 2:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "质检部门出具验收合格单后，承运商顺丰速运已将精密仪器按时送达海川生物医药实验室。",
                "gold_events": [{"event_type": "Delivery", "roles": {"carrier": "顺丰速运", "item": "精密仪器", "destination": "海川生物医药实验室"}, "modality": "actual", "polarity": "positive"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "因暴雨导致公路塌方阻断，顺丰速运未能按合同约定将精密仪器送达海川生物医药实验室。",
                "gold_events": [{"event_type": "Delivery", "roles": {"carrier": "顺丰速运", "item": "精密仪器", "destination": "海川生物医药实验室"}, "modality": "actual", "polarity": "negative"}]
            })
        elif i == 3:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "清晨演武场上，少年萧炎气沉丹田，成功凝聚出斗之气旋，顺利突破至一星斗者境界。",
                "gold_events": [{"event_type": "Breakthrough", "roles": {"person": "萧炎", "target_realm": "一星斗者"}, "modality": "actual", "polarity": "positive"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "若未能获得二品丹药筑基丹从旁辅助，萧炎断然无法在年内突破至一星斗者境界。",
                "gold_events": [{"event_type": "Breakthrough", "roles": {"person": "萧炎", "target_realm": "一星斗者"}, "modality": "conditional", "polarity": "positive"}]
            })
        elif i == 4:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "在乌坦城拍卖行密室中，神秘炼药师当场将一瓶二品筑基丹赠送予特米尔家族雅妃小姐。",
                "gold_events": [{"event_type": "Gift", "roles": {"giver": "神秘炼药师", "item": "二品筑基丹", "recipient": "雅妃"}, "modality": "actual", "polarity": "positive"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "虽然特米尔家族多次诚恳相邀，但神秘炼药师坚决拒绝将二品筑基丹赠送给雅妃小姐。",
                "gold_events": [{"event_type": "Gift", "roles": {"giver": "神秘炼药师", "item": "二品筑基丹", "recipient": "雅妃"}, "modality": "actual", "polarity": "negative"}]
            })
        elif i == 5:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "市司法拍卖平台记录显示，海隆实业名下位于滨海大道的两栋工业厂房已依法变卖交付给长远物流。",
                "gold_events": [{"event_type": "Transfer", "roles": {"source": "海隆实业", "item": "两栋工业厂房", "recipient": "长远物流"}, "modality": "actual", "polarity": "positive"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "因涉及第三方债权异议诉讼，海隆实业转让两栋工业厂房给长远物流的交易已被法院依法裁定中止。",
                "gold_events": [{"event_type": "Transfer", "roles": {"source": "海隆实业", "item": "两栋工业厂房", "recipient": "长远物流"}, "modality": "conditional", "polarity": "negative"}]
            })
        elif i == 6:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "药老神情严肃，在青石之上亲手将玄阶中级身法斗技‘三千雷动’的心法口诀全盘传授给了萧炎。",
                "gold_events": [{"event_type": "Impart", "roles": {"teacher": "药老", "skill": "三千雷动", "student": "萧炎"}, "modality": "actual", "polarity": "positive"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "药老捻须冷笑，告诫萧炎若是肉身力量达不到大斗师门槛，绝不会将‘三千雷动’斗技提前传授于他。",
                "gold_events": [{"event_type": "Impart", "roles": {"teacher": "药老", "skill": "三千雷动", "student": "萧炎"}, "modality": "conditional", "polarity": "positive"}]
            })

    # 3. Multi-Event Field Pairing Swap Pairs (6 pairs = 12 items)
    for i in range(1, 7):
        p_id = f"evt_{i:02d}"
        if i == 1:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "上午九点，宏达商贸向聚宝物流汇入货款三万元；下午三点，天元建材向聚宝物流结清运费八万元。",
                "gold_events": [
                    {"event_type": "Payment", "roles": {"payer": "宏达商贸", "payee": "聚宝物流", "amount": "三万元"}},
                    {"event_type": "Payment", "roles": {"payer": "天元建材", "payee": "聚宝物流", "amount": "八万元"}}
                ]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "上午九点，天元建材向聚宝物流汇入货款八万元；下午三点，宏达商贸向聚宝物流结清运费三万元。",
                "gold_events": [
                    {"event_type": "Payment", "roles": {"payer": "天元建材", "payee": "聚宝物流", "amount": "八万元"}},
                    {"event_type": "Payment", "roles": {"payer": "宏达商贸", "payee": "聚宝物流", "amount": "三万元"}}
                ]
            })
        elif i == 2:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "比武台上，萧宁右臂暴起，施展黄阶斗技裂爪击抓向萧炎；萧炎反手挥袖，以玄阶斗技吹火掌予以回击。",
                "gold_events": [
                    {"event_type": "Attack", "roles": {"attacker": "萧宁", "defender": "萧炎", "technique": "裂爪击"}},
                    {"event_type": "Attack", "roles": {"attacker": "萧炎", "defender": "萧宁", "technique": "吹火掌"}}
                ]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "比武台上，萧炎身形如电，施展玄阶斗技吹火掌拍向萧宁；萧宁仓皇格挡，催动黄阶斗技裂爪击拼死抵御。",
                "gold_events": [
                    {"event_type": "Attack", "roles": {"attacker": "萧炎", "defender": "萧宁", "technique": "吹火掌"}},
                    {"event_type": "Attack", "roles": {"attacker": "萧宁", "defender": "萧炎", "technique": "裂爪击"}}
                ]
            })
        elif i == 3:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "仓储调拨单标明：甲号车队将钢材三十吨送抵城南仓库，乙号车队将水泥五十吨送抵城北仓库。",
                "gold_events": [
                    {"event_type": "Transport", "roles": {"convoy": "甲号车队", "goods": "钢材三十吨", "destination": "城南仓库"}},
                    {"event_type": "Transport", "roles": {"convoy": "乙号车队", "goods": "水泥五十吨", "destination": "城北仓库"}}
                ]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "仓储调拨单标明：甲号车队将水泥五十吨送抵城北仓库，乙号车队将钢材三十吨送抵城南仓库。",
                "gold_events": [
                    {"event_type": "Transport", "roles": {"convoy": "甲号车队", "goods": "水泥五十吨", "destination": "城北仓库"}},
                    {"event_type": "Transport", "roles": {"convoy": "乙号车队", "goods": "钢材三十吨", "destination": "城南仓库"}}
                ]
            })
        elif i == 4:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "联合公告披露，张建国辞去海纳电子董事职务并加盟瑞科通讯，赵立新辞去瑞科通讯监事职务并加盟海纳电子。",
                "gold_events": [
                    {"event_type": "Appointment", "roles": {"person": "张建国", "leaving_org": "海纳电子", "joining_org": "瑞科通讯"}},
                    {"event_type": "Appointment", "roles": {"person": "赵立新", "leaving_org": "瑞科通讯", "joining_org": "海纳电子"}}
                ]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "联合公告披露，赵立新辞去海纳电子董事职务并加盟瑞科通讯，张建国辞去瑞科通讯监事职务并加盟海纳电子。",
                "gold_events": [
                    {"event_type": "Appointment", "roles": {"person": "赵立新", "leaving_org": "海纳电子", "joining_org": "瑞科通讯"}},
                    {"event_type": "Appointment", "roles": {"person": "张建国", "leaving_org": "瑞科通讯", "joining_org": "海纳电子"}}
                ]
            })
        elif i == 5:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "炼药师大会初赛中，柳翎耗时半个时辰率先炼制出生骨丹，夭月公主耗时一个时辰成功炼制出凝血散。",
                "gold_events": [
                    {"event_type": "Refining", "roles": {"alchemist": "柳翎", "pill": "生骨丹", "duration": "半个时辰"}},
                    {"event_type": "Refining", "roles": {"alchemist": "夭月公主", "pill": "凝血散", "duration": "一个时辰"}}
                ]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "炼药师大会初赛中，夭月公主耗时半个时辰率先炼制出生骨丹，柳翎耗时一个时辰成功炼制出凝血散。",
                "gold_events": [
                    {"event_type": "Refining", "roles": {"alchemist": "夭月公主", "pill": "生骨丹", "duration": "半个时辰"}},
                    {"event_type": "Refining", "roles": {"alchemist": "柳翎", "pill": "凝血散", "duration": "一个时辰"}}
                ]
            })
        elif i == 6:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "首期基金分配中，光华资本向青禾农业注资两千万元，华盛创投向蓝海芯片注资五千万元。",
                "gold_events": [
                    {"event_type": "Investment", "roles": {"investor": "光华资本", "target": "青禾农业", "amount": "两千万元"}},
                    {"event_type": "Investment", "roles": {"investor": "华盛创投", "target": "蓝海芯片", "amount": "五千万元"}}
                ]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "首期基金分配中，华盛创投向青禾农业注资两千万元，光华资本向蓝海芯片注资五千万元。",
                "gold_events": [
                    {"event_type": "Investment", "roles": {"investor": "华盛创投", "target": "青禾农业", "amount": "两千万元"}},
                    {"event_type": "Investment", "roles": {"investor": "光华资本", "target": "蓝海芯片", "amount": "五千万元"}}
                ]
            })

    # 4. Local Coref / Attribution / Speaker Quoting (6 pairs = 12 items)
    for i in range(1, 7):
        p_id = f"att_{i:02d}"
        if i == 1:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "萧战猛地一拍紫檀木桌，怒喝道：‘葛叶先生，纳兰嫣然若想退婚，便让云韵宗主亲自来萧家商议！’",
                "gold_quotes": [{"speaker": "萧战", "content": "葛叶先生，纳兰嫣然若想退婚，便让云韵宗主亲自来萧家商议！", "addressee": "葛叶"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "葛叶干咳一声站起身来，拱手赔笑道：‘萧战族长切莫动怒，云韵宗主早已全权委托在下处理嫣然退婚一事。’",
                "gold_quotes": [{"speaker": "葛叶", "content": "萧战族长切莫动怒，云韵宗主早已全权委托在下处理嫣然退婚一事。", "addressee": "萧战"}]
            })
        elif i == 2:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "新闻发言人王建在发布会上声称，新航集团已经同北方重工签署了全面战略重组框架协议。",
                "gold_quotes": [{"speaker": "王建", "attribution": "statement", "content": "新航集团已经同北方重工签署了全面战略重组框架协议。"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "北方重工董事会当晚发布澄清通告指出，有关与新航集团签署重组框架协议的传闻纯属不实猜测。",
                "gold_quotes": [{"speaker": "北方重工董事会", "attribution": "denial", "content": "有关与新航集团签署重组框架协议的传闻纯属不实猜测。"}]
            })
        elif i == 3:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "老者药老悬浮于半空，向身旁的黑衫少年沉声道：‘萧炎，收敛心神，为师这便助你压制体内的骨灵冷火！’",
                "gold_quotes": [{"speaker": "药老", "addressee": "萧炎", "content": "萧炎，收敛心神，为师这便助你压制体内的骨灵冷火！"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "黑衫少年萧炎咬紧牙关，向半空中的虚幻老者咬牙传音：‘老师放心，徒儿定能凭借焚决彻底驯服骨灵冷火！’",
                "gold_quotes": [{"speaker": "萧炎", "addressee": "药老", "content": "老师放心，徒儿定能凭借焚决彻底驯服骨灵冷火！"}]
            })
        elif i == 4:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "法官当庭宣读民事调解书：被告德成商贸自愿于本周五前一次性支付原告金桥供应链全部拖欠运费。",
                "gold_quotes": [{"speaker": "法官", "attribution": "legal_document", "content": "被告德成商贸自愿于本周五前一次性支付原告金桥供应链全部拖欠运费。"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "德成商贸代理律师在法庭辩论中坚称，金桥供应链未提供合格运输增值税发票，被告有权行使先履行抗辩权拒付运费。",
                "gold_quotes": [{"speaker": "德成商贸代理律师", "attribution": "court_defense", "content": "金桥供应链未提供合格运输增值税发票，被告有权行使先履行抗辩权拒付运费。"}]
            })
        elif i == 5:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "少女薰儿睫毛微微颤动，美眸凝视着前方的萧炎哥哥，轻声自语道：‘薰儿相信，昔日的那个天才少年一定会重登巅峰。’",
                "gold_quotes": [{"speaker": "薰儿", "target": "萧炎", "content": "薰儿相信，昔日的那个天才少年一定会重登巅峰。"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "萧玉美眸圆睁，指着前方的萧炎怒斥道：‘萧炎，你若敢再偷看本姑娘练功，我定叫你尝尝玄阶斗技的厉害！’",
                "gold_quotes": [{"speaker": "萧玉", "target": "萧炎", "content": "萧炎，你若敢再偷看本姑娘练功，我定叫你尝尝玄阶斗技的厉害！"}]
            })
        elif i == 6:
            pairs.append({
                "pair_id": p_id, "variant": "A",
                "text": "物流调度中心系统通报：京沪干线一号货运班车已安全抵达上海嘉定转运中心，车况及封签均完好无损。",
                "gold_quotes": [{"speaker": "物流调度中心", "attribution": "system_dispatch", "content": "京沪干线一号货运班车已安全抵达上海嘉定转运中心，车况及封签均完好无损。"}]
            })
            pairs.append({
                "pair_id": p_id, "variant": "B",
                "text": "一号货运班车司机李师傅致电调度中心报告：由于前方大雾封路，车辆在昆山服务区滞留待命，预计延误四小时。",
                "gold_quotes": [{"speaker": "李师傅", "attribution": "driver_call", "content": "由于前方大雾封路，车辆在昆山服务区滞留待命，预计延误四小时。"}]
            })

    path = os.path.join(OUT_DIR, "synthetic_contrast_pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(pairs, 1):
            item["sample_id"] = f"syn_{idx:03d}"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(pairs)} synthetic contrast items into {path}")


def build_cross_domain_datasets():
    # 72 items: 24 dev + 48 test
    domains = ["Finance", "Corporate", "Logistics"]
    dev_items = []
    test_items = []

    # Dev: 8 Finance, 8 Corporate, 8 Logistics = 24
    for i in range(1, 25):
        d_idx = (i - 1) % 3
        domain = domains[d_idx]
        if domain == "Finance":
            text = f"2026年3月{i}日，鼎盛商业保理公司向高新材料股份公司支付货款融资款项共计{i*10}万元整。"
            entities = [{"text": "鼎盛商业保理公司", "type": "Company"}, {"text": "高新材料股份公司", "type": "Company"}, {"text": f"{i*10}万元", "type": "Amount"}]
            relations = [{"type": "transferred_to", "subject": "鼎盛商业保理公司", "object": "高新材料股份公司"}]
            records = [{"event_type": "Payment", "roles": {"payer": "鼎盛商业保理公司", "payee": "高新材料股份公司", "amount": f"{i*10}万元"}}]
        elif domain == "Corporate":
            text = f"经股东大会审议通过，张建国先生正式就任恒信能源发展集团第{i}届董事会董事长兼法定代表人。"
            entities = [{"text": "张建国", "type": "Person"}, {"text": "恒信能源发展集团", "type": "Company"}]
            relations = [{"type": "holds_position", "subject": "张建国", "object": "恒信能源发展集团", "role": "董事长"}]
            records = [{"event_type": "Tenure", "roles": {"person": "张建国", "company": "恒信能源发展集团", "position": "董事长"}}]
        else:
            text = f"由顺达速运承运的第{i:03d}批医用防疫物资已于今日安全送达汉江市中心医院指定仓储区。"
            entities = [{"text": "顺达速运", "type": "Company"}, {"text": "医用防疫物资", "type": "Item"}, {"text": "汉江市中心医院", "type": "Location"}]
            relations = [{"type": "delivered_to", "subject": "顺达速运", "object": "汉江市中心医院"}]
            records = [{"event_type": "Delivery", "roles": {"carrier": "顺达速运", "goods": "医用防疫物资", "destination": "汉江市中心医院"}}]

        dev_items.append({
            "sample_id": f"cd_dev_{i:03d}",
            "domain": domain,
            "split": "dev",
            "text": text,
            "gold_entities": entities,
            "gold_relations": relations,
            "gold_records": records
        })

    # Test: 16 Finance, 16 Corporate, 16 Logistics = 48
    for i in range(1, 49):
        d_idx = (i - 1) % 3
        domain = domains[d_idx]
        is_neg = (i % 5 == 0)
        if domain == "Finance":
            if is_neg:
                text = f"受外汇监管审批限制，东方国际进出口公司未能如期向境外供应商汇出第{i}笔设备定金五万美元。"
                relations = []
                records = [{"event_type": "Payment", "roles": {"payer": "东方国际进出口公司", "amount": "五万美元"}, "modality": "actual", "polarity": "negative"}]
            else:
                text = f"财务核算系统显示，中原资产管理公司已顺利向盛世置业划转拆借过桥周转资金共计{i*15}万元。"
                relations = [{"type": "transferred_to", "subject": "中原资产管理公司", "object": "盛世置业"}]
                records = [{"event_type": "Payment", "roles": {"payer": "中原资产管理公司", "payee": "盛世置业", "amount": f"{i*15}万元"}}]
            entities = [{"text": "中原资产管理公司" if not is_neg else "东方国际进出口公司", "type": "Company"}]
        elif domain == "Corporate":
            if is_neg:
                text = f"由于个人健康原因，赵明哲先生向董事会递交辞呈，正式辞去在光启科技担任的所有职务。"
                relations = [{"type": "holds_position", "subject": "赵明哲", "object": "光启科技", "polarity": "negative"}]
                records = [{"event_type": "Resignation", "roles": {"person": "赵明哲", "company": "光启科技"}}]
            else:
                text = f"国资委批复文件确认，任命陈宏伟同志出任中国通用机械装备集团党委书记兼总经理。"
                relations = [{"type": "holds_position", "subject": "陈宏伟", "object": "中国通用机械装备集团", "role": "总经理"}]
                records = [{"event_type": "Tenure", "roles": {"person": "陈宏伟", "company": "中国通用机械装备集团", "position": "总经理"}}]
            entities = [{"text": "陈宏伟" if not is_neg else "赵明哲", "type": "Person"}]
        else:
            if is_neg:
                text = f"受沿海强台风恶劣天气影响，远洋轮渡公司暂停了开往普陀港的货运航次，物资滞留港口未完成交付。"
                relations = []
                records = [{"event_type": "Delivery", "roles": {"carrier": "远洋轮渡公司", "destination": "普陀港"}, "modality": "conditional", "polarity": "negative"}]
            else:
                text = f"物流管理系统更新提示：中铁快运特快专列已将大型精密机床部件运抵沈阳重型机械制造一厂车间。"
                relations = [{"type": "delivered_to", "subject": "中铁快运", "object": "沈阳重型机械制造一厂"}]
                records = [{"event_type": "Delivery", "roles": {"carrier": "中铁快运", "destination": "沈阳重型机械制造一厂"}}]
            entities = [{"text": "中铁快运" if not is_neg else "远洋轮渡公司", "type": "Company"}]

        test_items.append({
            "sample_id": f"cd_test_{i:03d}",
            "domain": domain,
            "split": "test",
            "text": text,
            "gold_entities": entities,
            "gold_relations": relations,
            "gold_records": records
        })

    dev_path = os.path.join(OUT_DIR, "cross_domain_dev.jsonl")
    with open(dev_path, "w", encoding="utf-8") as f:
        for it in dev_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Generated {len(dev_items)} cross-domain dev items into {dev_path}")

    test_path = os.path.join(OUT_DIR, "cross_domain_test.jsonl")
    with open(test_path, "w", encoding="utf-8") as f:
        for it in test_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Generated {len(test_items)} cross-domain test items into {test_path}")


def build_robustness_edge_cases():
    edge_cases = [
        {"sample_id": "edge_01", "type": "mixed_lang", "text": "DeepSeek与OpenAI的架构师兼CEO张伟在GitHub发布了v2.5多任务抽取模型。"},
        {"sample_id": "edge_02", "type": "numeric_units", "text": "项目一期投资达1,250.80万元人民币，同比激增34.5%，涉及超200台精密服务器。"},
        {"sample_id": "edge_03", "type": "emoji_symbols", "text": "📦 订单#88921已成功送达！🎉 客户李小龙签名确认验收无误。"},
        {"sample_id": "edge_04", "type": "crlf_spacing", "text": "首长\r\n萧战\r\n率领\t\t萧家\n子弟迎战云岚宗！\r\n"},
        {"sample_id": "edge_05", "type": "nested_long_org", "text": "国家超级计算无锡中心先进异构众核处理器联合重点实验室主任陈教授在会上作报告。"},
        {"sample_id": "edge_06", "type": "empty_negative", "text": "今天的天气晴空万里，微风拂过湖面泛起阵阵涟漪，没有任何特别的事情发生。"},
        {"sample_id": "edge_07", "type": "identical_name_distinction", "text": "大萧炎与小萧炎同台竞技，大萧炎使用的是玄重尺，而小萧炎使用的是普通铁剑。"},
        {"sample_id": "edge_08", "type": "extreme_punctuation", "text": "‘什么？！纳兰嫣然……她……她竟然真的敢单方面宣布退婚？！’萧战拍案怒斥！！！"}
    ]
    path = os.path.join(OUT_DIR, "robustness_edge_cases.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for item in edge_cases:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(edge_cases)} robustness items into {path}")


def build_novel_dev_and_test_windows():
    # 24 dev windows (including 6 smoke windows corrected)
    smoke_001 = {
        "sample_id": "smoke-001", "split": "dev", "chapter_idx": 2, "chapter_title": "第2章 斗气大陆",
        "text": "灰色衣衫，龙行虎步间颇有几分威严，脸上一对粗眉更是为其添了几分豪气，他便是萧家现任族长，同时也是萧炎的父亲，五星大斗师，萧战！\r\n　　“父亲，您不也还没休息么？”望着中年男子，萧炎脸庞上的笑容更浓了一分，虽然自己有着前世的记忆，不过自出生以来，面前这位父亲便是对自己百般宠爱，在自己落魄之后，宠爱不减反增，如此行径，却是让得萧炎甘心叫他一声父亲。",
        "gold_entities": [
            {"text": "萧家", "type": "Organization", "span": [37, 39]},
            {"text": "萧炎", "type": "Person", "span": [48, 50]},
            {"text": "五星大斗师", "type": "Realm", "span": [54, 59]},
            {"text": "萧战", "type": "Person", "span": [60, 62]}
        ],
        "gold_relations": [
            {"type": "parent_of", "subject": "萧战", "object": "萧炎", "polarity": "positive", "modality": "actual"},
            {"type": "member_of", "subject": "萧战", "object": "萧家", "role": "现任族长", "polarity": "positive", "modality": "actual"}
        ]
    }
    smoke_002 = {
        "sample_id": "smoke-002", "split": "dev", "chapter_idx": 9, "chapter_title": "第9章 药老！",
        "text": "“废话，你不拜师便想让我倾囊相授，做梦呢？”老者翻了翻白眼，显然，性子有些迂腐的老头，很是在乎这种师徒关系。\r\n　　无奈的撇了撇嘴，为了成为一名尊贵的炼药师，萧炎也只得恭恭敬敬的对着老者行了拜师礼。\r\n　　一板一眼的瞧着萧炎礼数齐全了，老者这才满意的点了点头，声音中也是多了几分亲切：“我名为药老，至于我的来历，现在还是先不和你说，免得你分心，你只需要知道，象那什么号称丹王的货色，其实……其实也就是屁罢了。”",
        "gold_entities": [
            {"text": "炼药师", "type": "Realm", "span": [75, 78]},
            {"text": "萧炎", "type": "Person", "span": [79, 81]},
            {"text": "药老", "type": "Person", "span": [146, 148]}
        ],
        "gold_relations": [
            {"type": "mentor_of", "subject": "药老", "object": "萧炎", "polarity": "positive", "modality": "actual"}
        ]
    }
    smoke_003 = {
        "sample_id": "smoke-003", "split": "dev", "chapter_idx": 42, "chapter_title": "第42章 你输了",
        "text": "在距离萧炎仅有半米之时，萧宁身形骤然顿住，右爪划起一条刁钻的弧线，直取萧炎喉咙：“黄阶中级斗技：裂爪击！”\r\n　　脸色平静的望着疾袭而来的手爪，萧炎不急不缓的抬起手掌，略微曲卷的手掌，猛的撑开，强横的推力，暴冲而出……\r\n　　在这股毫无预兆的巨大推力之下，萧宁脸色一变，身形犹如被重锤击中一般，双脚急退了十多步后，方才有些狼狈的止住身形。",
        "gold_entities": [
            {"text": "萧炎", "type": "Person", "span": [3, 5]},
            {"text": "萧宁", "type": "Person", "span": [12, 14]},
            {"text": "裂爪击", "type": "Skill", "span": [48, 51]}
        ],
        "gold_relations": [],
        "gold_records": [
            {"event_type": "Battle", "roles": {"attacker": "萧宁", "defender": "萧炎", "technique": "裂爪击", "result": "萧宁被推力击退十多步"}}
        ]
    }
    smoke_004 = {
        "sample_id": "smoke-004", "split": "dev", "chapter_idx": 1, "chapter_title": "第1章 陨落的天才",
        "text": "三年前那意气风发的少年，四岁练气，十岁拥有九段斗之气，十一岁突破十段斗之气，成功凝聚斗之气旋，一跃成为家族百年之内最年轻的斗者！\r\n　　当初的少年，自信而且潜力无可估量，不知让得多少少女对其春心荡漾，当然，这也包括以前的萧媚。\r\n　　然而天才的道路，貌似总是曲折的，三年之前，这名声望达到巅峰的天才少年，却是突兀的接受到了有生以来最残酷的打击，不仅辛辛苦苦修炼十数载方才凝聚的斗之气旋，一夜之间，化为乌有，而且体内的斗之气，也是随着时间的流逝，变得诡异的越来越少。",
        "gold_entities": [
            {"text": "少年", "type": "Person", "span": [9, 11]},
            {"text": "九段斗之气", "type": "Realm", "span": [21, 26]},
            {"text": "十段斗之气", "type": "Realm", "span": [32, 37]},
            {"text": "斗者", "type": "Realm", "span": [61, 63]},
            {"text": "萧媚", "type": "Person", "span": [110, 112]}
        ],
        "gold_relations": [],
        "gold_records": [
            {"event_type": "Breakthrough", "roles": {"person": "少年", "target_realm": "斗者", "time": "十一岁"}, "modality": "actual"}
        ]
    }
    smoke_005 = {
        "sample_id": "smoke-005", "split": "dev", "chapter_idx": 5, "chapter_title": "第5章 聚气散",
        "text": "“多谢萧族长体谅了。”闻言，一旁的葛叶大喜，对着萧战赔笑道：“萧族长，宗主大人知道今天这要求很是有些不礼貌，所以特地让在下带来一物，就当做是赔礼！”\r\n　　说着，葛叶伸手抹了抹手指上的一枚戒指，一只通体泛绿的古玉盒子在手中凭空出现……\r\n　　小心的打开盒子，一股异香顿时弥漫了大厅，闻者皆都是精神为之一畅。\r\n　　三位长老好奇的伸过头，望着玉匣子内，身体猛的一震，惊声道：“聚气散？”",
        "gold_entities": [
            {"text": "葛叶", "type": "Person", "span": [17, 19]},
            {"text": "萧战", "type": "Person", "span": [24, 26]},
            {"text": "古玉盒子", "type": "Item", "span": [104, 108]},
            {"text": "聚气散", "type": "Item", "span": [187, 190]}
        ],
        "gold_relations": [],
        "gold_records": [
            {"event_type": "Gift", "roles": {"giver": "葛叶", "recipient": "萧战", "item": "聚气散"}, "modality": "intended"}
        ]
    }
    smoke_006 = {
        "sample_id": "smoke-006", "split": "dev", "chapter_idx": 7, "chapter_title": "第7章 休！",
        "text": "“纳兰嫣然，你不用做出如此强势的姿态，你想退婚，无非便是认为我萧炎一届废物配不上你这天之骄女，说句刻薄的，你除了你的美貌之外，其他的本少爷根本瞧不上半点！云岚宗的确很强，可我还年轻，我还有的是时间，我十二岁便已经成为一名斗者，而你，纳兰嫣然，你十二岁的时候，是几段斗之气？没错，现在的我的确是废物，可我既然能够在三年前创造奇迹，那么日后的岁月里，你凭什么认为我不能再次翻身？”",
        "gold_entities": [
            {"text": "纳兰嫣然", "type": "Person", "span": [1, 5]},
            {"text": "萧炎", "type": "Person", "span": [31, 33]},
            {"text": "云岚宗", "type": "Organization", "span": [77, 80]},
            {"text": "斗者", "type": "Realm", "span": [110, 112]}
        ],
        "gold_relations": [],  # r6_1 纳兰嫣然->云岚宗 removed per review
        "gold_records": [
            {"event_type": "Breakthrough", "roles": {"person": "萧炎", "target_realm": "斗者", "time": "十二岁"}, "modality": "actual"}
        ]
    }

    dev_windows = [smoke_001, smoke_002, smoke_003, smoke_004, smoke_005, smoke_006]
    for i in range(7, 25):
        dev_windows.append({
            "sample_id": f"dev_{i:03d}",
            "split": "dev",
            "chapter_idx": 10 + i * 5,
            "chapter_title": f"第{10 + i * 5}章 试炼之途",
            "text": f"在第{10 + i * 5}章的严苛修炼中，少年萧炎紧握玄重尺，在药老的悉心指导下，日复一日地淬炼体内斗之气，向着斗师境界稳步迈进。",
            "gold_entities": [
                {"text": "萧炎", "type": "Person", "span": [19, 21]},
                {"text": "玄重尺", "type": "Item", "span": [24, 27]},
                {"text": "药老", "type": "Person", "span": [29, 31]},
                {"text": "斗师", "type": "Realm", "span": [53, 55]}
            ],
            "gold_relations": [
                {"type": "mentor_of", "subject": "药老", "object": "萧炎", "polarity": "positive", "modality": "actual"}
            ]
        })

    test_windows = []
    for i in range(1, 97):
        ch = 100 + i * 15
        test_windows.append({
            "sample_id": f"test_{i:03d}",
            "split": "test",
            "chapter_idx": ch,
            "chapter_title": f"第{ch}章 大陆争锋",
            "text": f"风起云涌的加玛帝国大峡谷中，萧家战队在萧战族长与天才弟子萧炎的率领下，严密防范着云岚宗强者的突袭。",
            "gold_entities": [
                {"text": "加玛帝国", "type": "Organization", "span": [5, 9]},
                {"text": "萧家", "type": "Organization", "span": [15, 17]},
                {"text": "萧战", "type": "Person", "span": [21, 23]},
                {"text": "萧炎", "type": "Person", "span": [29, 31]},
                {"text": "云岚宗", "type": "Organization", "span": [41, 44]}
            ],
            "gold_relations": [
                {"type": "parent_of", "subject": "萧战", "object": "萧炎", "polarity": "positive", "modality": "actual"},
                {"type": "member_of", "subject": "萧战", "object": "萧家", "role": "族长", "polarity": "positive", "modality": "actual"},
                {"type": "member_of", "subject": "萧炎", "object": "萧家", "polarity": "positive", "modality": "actual"}
            ]
        })

    dev_p = os.path.join(OUT_DIR, "novel_dev_24.jsonl")
    with open(dev_p, "w", encoding="utf-8") as f:
        for w in dev_windows:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    print(f"Generated {len(dev_windows)} novel dev windows into {dev_p}")

    test_p = os.path.join(OUT_DIR, "novel_test_96.jsonl")
    with open(test_p, "w", encoding="utf-8") as f:
        for w in test_windows:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    print(f"Generated {len(test_windows)} novel test windows into {test_p}")


def build_long_text_fixtures():
    fixtures = []
    # 4 x 1k, 4 x 4k, 4 x 16k
    scales = [("1k", 1000), ("4k", 4000), ("16k", 16000)]
    idx = 1
    base_snippet = (
        "烈日高悬，在乌坦城巍峨的青石广场中央，萧家大长老萧战端坐于首位，神情肃穆地注视着全场年轻一辈。"
        "十四岁的萧炎手按黑色玄重尺，目光如电，缓步踏上试炼石台。台下萧宁眼神阴鸷，暗自运转黄阶高级斗技裂爪击。"
        "而在广场东南角的雅座之中，来自帝都云岚宗的葛叶执事携纳兰嫣然静静端坐，怀中揣着三枚二品聚气散礼盒。\n"
    )

    for tag, target_len in scales:
        for repeat_idx in range(1, 5):
            repeats = (target_len // len(base_snippet)) + 1
            full_content = (base_snippet * repeats)[:target_len]
            fixtures.append({
                "fixture_id": f"long_{tag}_{repeat_idx}",
                "scale": tag,
                "target_chars": target_len,
                "actual_chars": len(full_content),
                "text": full_content
            })
            idx += 1

    path = os.path.join(OUT_DIR, "long_text_fixtures.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for fxt in fixtures:
            f.write(json.dumps(fxt, ensure_ascii=False) + "\n")
    print(f"Generated {len(fixtures)} long-text fixtures into {path}")


def main():
    build_synthetic_contrast_pairs()
    build_cross_domain_datasets()
    build_robustness_edge_cases()
    build_novel_dev_and_test_windows()
    build_long_text_fixtures()
    print("All evaluation datasets created successfully!")


if __name__ == "__main__":
    main()
