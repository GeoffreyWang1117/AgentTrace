#!/usr/bin/env python3
"""
AgentTrace 功能测试脚本
测试所有核心功能并保存日志
"""

import json
import requests
from datetime import datetime

BASE_URL = "http://127.0.0.1:8000"
LOG_FILE = "logs/agenttrace_demo.log"

def log(msg, data=None):
    """写入日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"[{timestamp}] {msg}\n")
        f.write(f"{'='*60}\n")
        if data:
            if isinstance(data, dict):
                f.write(json.dumps(data, indent=2, ensure_ascii=False))
            else:
                f.write(str(data))
            f.write("\n")
    print(f"[{timestamp}] {msg}")

def main():
    # 初始化日志
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AgentTrace 多智能体因果追踪系统 - 功能演示日志\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")

    # 1. 健康检查
    log("1. 系统健康检查")
    resp = requests.get(f"{BASE_URL}/health")
    log("健康状态", resp.json())

    # 2. 列出所有运行
    log("2. 获取所有追踪运行列表")
    resp = requests.get(f"{BASE_URL}/api/runs")
    runs_data = resp.json()
    log("运行列表", runs_data)

    if not runs_data["runs"]:
        log("错误: 没有可用的运行记录")
        return

    # 选择第一个运行进行分析
    run_id = runs_data["runs"][0]["run_id"]
    log(f"3. 选择运行 {run_id} 进行详细分析")

    # 3. 获取运行统计
    log("4. 获取运行统计信息")
    resp = requests.get(f"{BASE_URL}/api/runs/{run_id}")
    log("统计信息", resp.json())

    # 4. 获取完整因果图
    log("5. 获取完整因果追踪图")
    resp = requests.get(f"{BASE_URL}/api/runs/{run_id}/graph")
    graph_data = resp.json()
    log("因果图概要", {
        "run_id": graph_data["run_id"],
        "node_count": len(graph_data["nodes"]),
        "edge_count": len(graph_data["edges"]),
        "statistics": graph_data["statistics"]
    })

    # 保存前10个节点作为示例
    log("示例节点 (前5个)", graph_data["nodes"][:5])

    # 5. 获取错误列表
    log("6. 获取错误节点列表")
    resp = requests.get(f"{BASE_URL}/api/runs/{run_id}/errors")
    errors_data = resp.json()
    log("错误列表", errors_data)

    if errors_data["errors"]:
        error_id = errors_data["errors"][0]["id"]

        # 6. 错误根因分析
        log(f"7. 错误根因分析 (错误ID: {error_id[:16]}...)")
        resp = requests.get(f"{BASE_URL}/api/runs/{run_id}/analyze-error/{error_id}")
        analysis = resp.json()
        log("错误分析结果", {
            "error_info": analysis["error_node"]["data"],
            "root_causes_count": len(analysis["root_causes"]),
            "causal_chain_length": len(analysis["causal_chains"][0]) if analysis["causal_chains"] else 0,
            "likely_causes": [
                {"score": c["score"], "reasons": c["reasons"]}
                for c in analysis["likely_causes"][:3]
            ]
        })

        # 7. 后向追踪
        log(f"8. 后向追踪 - 追溯错误原因")
        resp = requests.post(
            f"{BASE_URL}/api/runs/{run_id}/trace/backward",
            json={"node_id": error_id, "max_depth": 5}
        )
        backward_data = resp.json()
        log("后向追踪结果", {
            "target_node_id": backward_data["target_node_id"][:16] + "...",
            "cause_count": backward_data["count"],
            "causes": [
                {"type": n["type"], "agent": n["agent_id"]}
                for n in backward_data["cause_nodes"][:5]
            ]
        })

        # 8. 获取根因
        log("9. 获取根本原因 (Root Causes)")
        resp = requests.get(f"{BASE_URL}/api/runs/{run_id}/root-causes/{error_id}")
        root_causes = resp.json()
        log("根本原因", root_causes)

        # 9. 获取因果链
        log("10. 获取完整因果链")
        resp = requests.get(f"{BASE_URL}/api/runs/{run_id}/causal-chains/{error_id}")
        chains = resp.json()
        log("因果链", {
            "chain_count": chains["chain_count"],
            "first_chain": [
                {"type": n["type"], "agent": n["agent_id"], "data_preview": str(n["data"])[:50]}
                for n in chains["chains"][0]
            ] if chains["chains"] else []
        })

    # 10. 前向追踪 - 从第一个输入节点开始
    input_nodes = [n for n in graph_data["nodes"] if n["type"] == "agent_input"]
    if input_nodes:
        first_input = input_nodes[0]
        log(f"11. 前向追踪 - 追踪输入影响 (节点: {first_input['id'][:16]}...)")
        resp = requests.post(
            f"{BASE_URL}/api/runs/{run_id}/trace/forward",
            json={"node_id": first_input["id"], "max_depth": 10}
        )
        forward_data = resp.json()
        log("前向追踪结果", {
            "source_node_id": forward_data["source_node_id"][:16] + "...",
            "affected_count": forward_data["count"],
            "affected_types": list(set(n["type"] for n in forward_data["affected_nodes"]))
        })

    # 11. 反事实分析
    tool_calls = [n for n in graph_data["nodes"] if n["type"] == "tool_call"]
    if tool_calls:
        tool_node = tool_calls[0]
        log(f"12. 反事实分析 - What-If 场景")
        log(f"    原始工具调用: {tool_node['data']}")

        alternative_data = dict(tool_node["data"])
        if "args" in alternative_data and alternative_data["args"]:
            # 修改第一个参数
            alternative_data["args"] = [99.99] + alternative_data["args"][1:]

        resp = requests.post(
            f"{BASE_URL}/api/runs/{run_id}/counterfactual",
            json={
                "node_id": tool_node["id"],
                "alternative_data": alternative_data
            }
        )
        cf_result = resp.json()
        log("反事实分析结果", {
            "original_run": cf_result["original_run_id"],
            "counterfactual_run": cf_result["counterfactual_run_id"],
            "comparison": cf_result["comparison"]
        })

    # 12. 路径查找
    if len(graph_data["nodes"]) >= 2:
        node1 = graph_data["nodes"][0]["id"]
        node2 = graph_data["nodes"][-1]["id"]
        log(f"13. 路径查找 - 两节点间因果路径")
        resp = requests.get(
            f"{BASE_URL}/api/runs/{run_id}/path",
            params={"source_id": node1, "target_id": node2}
        )
        path_data = resp.json()
        log("路径查找结果", {
            "path_exists": path_data["exists"],
            "path_length": path_data.get("length", 0)
        })

    # 13. 推断新的因果关系
    log("14. 运行因果关系推断引擎")
    resp = requests.post(f"{BASE_URL}/api/runs/{run_id}/infer")
    infer_result = resp.json()
    log("推断结果", {
        "inferred_edges_count": infer_result["count"],
        "sample_edges": [
            {
                "type": e["type"],
                "confidence": e["confidence"],
                "method": e["metadata"].get("inference_method", "unknown")
            }
            for e in infer_result["inferred_edges"][:3]
        ] if infer_result["inferred_edges"] else []
    })

    # 总结
    log("=" * 40)
    log("测试完成总结")
    log("=" * 40)
    summary = {
        "总运行数": runs_data["total"],
        "分析的运行": run_id,
        "节点总数": graph_data["statistics"]["node_count"],
        "边总数": graph_data["statistics"]["edge_count"],
        "参与的Agent": graph_data["statistics"]["agents"],
        "节点类型分布": graph_data["statistics"]["node_types"],
        "错误数量": errors_data["count"],
        "是否为DAG": graph_data["statistics"]["is_dag"]
    }
    log("系统统计", summary)

    print(f"\n日志已保存到: {LOG_FILE}")


if __name__ == "__main__":
    main()
