"""
Expand Benchmark to 500+ Scenarios.

Generates additional scenarios to create a larger benchmark:
1. More diversity in each existing domain
2. New domains: Healthcare, Legal, Education
3. More bug type variations
4. Longer causal chains
"""

import json
import random
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.benchmark.scenario_generator import (
    ScenarioGenerator,
    ScenarioDomain,
    BugType
)


class ExtendedScenarioGenerator(ScenarioGenerator):
    """Extended scenario generator with more domains and variations."""

    # New domains
    HEALTHCARE_AGENTS = [
        {"id": "agent_1", "role": "Triage Nurse", "capabilities": ["assess_symptoms", "prioritize_patients"]},
        {"id": "agent_2", "role": "Diagnostician", "capabilities": ["analyze_symptoms", "suggest_tests"]},
        {"id": "agent_3", "role": "Lab Technician", "capabilities": ["run_tests", "report_results"]},
        {"id": "agent_4", "role": "Physician", "capabilities": ["make_diagnosis", "prescribe_treatment"]},
        {"id": "agent_5", "role": "Pharmacist", "capabilities": ["verify_prescription", "dispense_medication"]}
    ]

    LEGAL_AGENTS = [
        {"id": "agent_1", "role": "Paralegal", "capabilities": ["research_cases", "prepare_documents"]},
        {"id": "agent_2", "role": "Attorney", "capabilities": ["analyze_case", "develop_strategy"]},
        {"id": "agent_3", "role": "Witness Coordinator", "capabilities": ["contact_witnesses", "schedule_depositions"]},
        {"id": "agent_4", "role": "Court Clerk", "capabilities": ["file_motions", "track_deadlines"]},
        {"id": "agent_5", "role": "Judge Assistant", "capabilities": ["review_filings", "schedule_hearings"]}
    ]

    EDUCATION_AGENTS = [
        {"id": "agent_1", "role": "Curriculum Designer", "capabilities": ["design_lessons", "set_objectives"]},
        {"id": "agent_2", "role": "Content Creator", "capabilities": ["create_materials", "develop_exercises"]},
        {"id": "agent_3", "role": "Tutor", "capabilities": ["teach_concepts", "answer_questions"]},
        {"id": "agent_4", "role": "Assessor", "capabilities": ["create_tests", "grade_submissions"]},
        {"id": "agent_5", "role": "Student Advisor", "capabilities": ["track_progress", "provide_feedback"]}
    ]

    FINANCE_AGENTS = [
        {"id": "agent_1", "role": "Data Analyst", "capabilities": ["collect_data", "analyze_trends"]},
        {"id": "agent_2", "role": "Risk Assessor", "capabilities": ["evaluate_risk", "calculate_exposure"]},
        {"id": "agent_3", "role": "Portfolio Manager", "capabilities": ["allocate_assets", "rebalance_portfolio"]},
        {"id": "agent_4", "role": "Compliance Officer", "capabilities": ["check_regulations", "approve_trades"]},
        {"id": "agent_5", "role": "Report Generator", "capabilities": ["create_reports", "notify_stakeholders"]}
    ]

    DEVOPS_AGENTS = [
        {"id": "agent_1", "role": "Build Agent", "capabilities": ["compile_code", "run_tests"]},
        {"id": "agent_2", "role": "Deploy Agent", "capabilities": ["deploy_artifacts", "configure_services"]},
        {"id": "agent_3", "role": "Monitor Agent", "capabilities": ["collect_metrics", "detect_anomalies"]},
        {"id": "agent_4", "role": "Alert Agent", "capabilities": ["send_notifications", "escalate_issues"]},
        {"id": "agent_5", "role": "Rollback Agent", "capabilities": ["revert_changes", "restore_backups"]}
    ]

    NEW_DOMAINS = {
        'healthcare': {
            'agents': HEALTHCARE_AGENTS,
            'task_templates': [
                "Process patient {patient_id} presenting with {symptoms}",
                "Emergency triage for incoming patients in {department}",
                "Coordinate care for patient with {condition}",
                "Process prescription refill request for {medication}",
                "Schedule follow-up care for post-operative patient"
            ],
            'actions': [
                ('triage', 'assess_patient'),
                ('diagnose', 'analyze_symptoms'),
                ('test', 'run_lab_tests'),
                ('treat', 'prescribe_medication'),
                ('dispense', 'provide_medication'),
                ('followup', 'schedule_appointment')
            ]
        },
        'legal': {
            'agents': LEGAL_AGENTS,
            'task_templates': [
                "Prepare case materials for {case_type} matter",
                "Research precedents for {legal_issue}",
                "Coordinate discovery process for {case_name}",
                "File motion for {motion_type} in {court}",
                "Prepare for deposition of {witness_name}"
            ],
            'actions': [
                ('research', 'find_precedents'),
                ('draft', 'prepare_document'),
                ('review', 'analyze_filing'),
                ('file', 'submit_to_court'),
                ('coordinate', 'schedule_meeting'),
                ('notify', 'inform_parties')
            ]
        },
        'education': {
            'agents': EDUCATION_AGENTS,
            'task_templates': [
                "Develop curriculum for {subject} course",
                "Create learning module on {topic}",
                "Assess student progress in {course_name}",
                "Provide tutoring session for {concept}",
                "Generate progress report for {student_group}"
            ],
            'actions': [
                ('design', 'create_curriculum'),
                ('develop', 'create_content'),
                ('teach', 'deliver_lesson'),
                ('assess', 'evaluate_understanding'),
                ('feedback', 'provide_guidance'),
                ('report', 'summarize_progress')
            ]
        },
        'finance': {
            'agents': FINANCE_AGENTS,
            'task_templates': [
                "Analyze market data for {asset_class}",
                "Assess risk exposure for {portfolio}",
                "Rebalance portfolio based on {strategy}",
                "Generate compliance report for {period}",
                "Process trade request for {security}"
            ],
            'actions': [
                ('analyze', 'process_data'),
                ('assess', 'evaluate_risk'),
                ('trade', 'execute_order'),
                ('verify', 'check_compliance'),
                ('report', 'generate_summary'),
                ('notify', 'alert_stakeholders')
            ]
        },
        'devops': {
            'agents': DEVOPS_AGENTS,
            'task_templates': [
                "Deploy version {version} to {environment}",
                "Process CI/CD pipeline for {repository}",
                "Monitor service health for {service_name}",
                "Respond to incident {incident_id}",
                "Rollback deployment due to {issue}"
            ],
            'actions': [
                ('build', 'compile_artifacts'),
                ('test', 'run_test_suite'),
                ('deploy', 'push_to_environment'),
                ('monitor', 'check_metrics'),
                ('alert', 'send_notification'),
                ('rollback', 'revert_deployment')
            ]
        }
    }

    def generate_new_domain_scenario(
        self,
        domain: str,
        bug_type: str
    ) -> Dict[str, Any]:
        """Generate scenario for new domains."""
        if domain not in self.NEW_DOMAINS:
            raise ValueError(f"Unknown domain: {domain}")

        domain_config = self.NEW_DOMAINS[domain]
        agents = domain_config['agents']
        task_template = random.choice(domain_config['task_templates'])
        actions = domain_config['actions']

        # Generate unique ID
        scenario_id = f"{domain[:3]}_{uuid.uuid4().hex[:6]}"

        # Fill template with random values
        task_description = task_template.format(
            patient_id=f"P{random.randint(1000,9999)}",
            symptoms="fever and cough",
            department="Emergency",
            condition="diabetes",
            medication="metformin",
            case_type="contract dispute",
            legal_issue="intellectual property",
            case_name="Smith v. Jones",
            motion_type="summary judgment",
            court="District Court",
            witness_name="John Doe",
            subject="Mathematics",
            topic="Calculus fundamentals",
            course_name="Algebra 101",
            concept="quadratic equations",
            student_group="Fall 2025 cohort",
            asset_class="equities",
            portfolio="growth fund",
            strategy="momentum",
            period="Q4 2025",
            security="AAPL",
            version="2.1.0",
            environment="production",
            repository="main-app",
            service_name="api-gateway",
            incident_id=f"INC{random.randint(1000,9999)}",
            issue="high error rate"
        )

        # Generate flow
        num_steps = random.randint(5, 8)
        bug_step = random.randint(2, num_steps - 2)
        error_step = random.randint(bug_step + 1, num_steps)

        buggy_flow = []
        for step in range(1, num_steps + 1):
            action_idx = (step - 1) % len(actions)
            agent_idx = step % len(agents)

            action_name, action_type = actions[action_idx]

            step_data = {
                "step": step,
                "from_agent": agents[agent_idx]["id"],
                "to_agent": agents[(agent_idx + 1) % len(agents)]["id"],
                "action": f"{action_name}_{action_type}",
                "content": {"data": f"result_step_{step}"},
                "is_bug_point": step == bug_step,
                "bug_description": self._get_bug_description(bug_type, action_type) if step == bug_step else None
            }
            buggy_flow.append(step_data)

        return {
            "scenario_id": scenario_id,
            "domain": domain,
            "complexity": "medium",
            "bug_type": bug_type,
            "task_description": task_description,
            "expected_outcome": f"Successful completion of {domain} workflow",
            "agents": agents[:num_steps],
            "buggy_flow": buggy_flow,
            "bug_injection_point": bug_step,
            "root_cause_step": bug_step,
            "error_manifestation_step": error_step,
            "causal_chain": [
                {"source_step": i, "target_step": i+1, "relationship_type": "data_flow"}
                for i in range(bug_step, error_step)
            ],
            "created_at": datetime.now().isoformat()
        }

    def _get_bug_description(self, bug_type: str, action: str) -> str:
        """Generate bug description."""
        descriptions = {
            "data_corruption": f"Data corrupted during {action}",
            "missing_context": f"Required context missing in {action}",
            "invalid_output": f"Invalid output format from {action}",
            "logic_error": f"Logic error in {action} processing",
            "state_inconsistency": f"State became inconsistent after {action}",
            "timeout_cascade": f"Timeout in {action} caused cascade",
            "race_condition": f"Race condition during {action}",
            "wrong_routing": f"Incorrect routing after {action}"
        }
        return descriptions.get(bug_type, f"Error in {action}")


def expand_benchmark(target_count: int = 500):
    """Expand benchmark to target count."""
    print("="*70)
    print(f"EXPANDING BENCHMARK TO {target_count}+ SCENARIOS")
    print("="*70)

    scenarios_dir = Path("data/scenarios")
    extended_generator = ExtendedScenarioGenerator()

    # Count existing scenarios
    existing_count = 0
    for domain_dir in scenarios_dir.iterdir():
        if domain_dir.is_dir():
            existing_count += len(list(domain_dir.glob("*.json")))

    print(f"\nExisting scenarios: {existing_count}")
    needed = target_count - existing_count

    if needed <= 0:
        print(f"Already have {existing_count} scenarios, no expansion needed.")
        return

    print(f"Need to generate: {needed} more scenarios")

    # Distribution across domains
    all_domains = list(ScenarioDomain) + list(ExtendedScenarioGenerator.NEW_DOMAINS.keys())
    bug_types = [bt.value for bt in BugType]

    generated = 0
    domain_counts = {d: 0 for d in all_domains}

    # Generate for new domains first
    print("\n--- Generating New Domain Scenarios ---")
    new_domains = list(ExtendedScenarioGenerator.NEW_DOMAINS.keys())

    for domain in new_domains:
        domain_dir = scenarios_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        # Generate ~50 scenarios per new domain
        for _ in range(50):
            if generated >= needed:
                break

            bug_type = random.choice(bug_types)

            try:
                scenario = extended_generator.generate_new_domain_scenario(domain, bug_type)

                # Save scenario
                scenario_file = domain_dir / f"{scenario['scenario_id']}.json"
                with open(scenario_file, 'w') as f:
                    json.dump(scenario, f, indent=2)

                generated += 1
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                if generated % 25 == 0:
                    print(f"  Generated {generated}/{needed} scenarios...")

            except Exception as e:
                print(f"  Error generating scenario: {e}")

    # Generate more for existing domains
    print("\n--- Expanding Existing Domains ---")
    existing_domains = [d.value for d in ScenarioDomain]

    for domain_enum in ScenarioDomain:
        domain = domain_enum.value
        domain_dir = scenarios_dir / domain

        # Generate additional scenarios
        for _ in range(30):  # 30 more per existing domain
            if generated >= needed:
                break

            bug_type = random.choice([bt for bt in BugType])

            try:
                scenario = extended_generator.generate_scenario(domain_enum, bug_type)

                # Save scenario
                scenario_file = domain_dir / f"{scenario['scenario_id']}.json"
                with open(scenario_file, 'w') as f:
                    json.dump(scenario, f, indent=2)

                generated += 1
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

            except Exception as e:
                continue

    # Final count
    final_count = 0
    print("\n--- Final Statistics ---")
    for domain_dir in scenarios_dir.iterdir():
        if domain_dir.is_dir():
            count = len(list(domain_dir.glob("*.json")))
            final_count += count
            print(f"  {domain_dir.name}: {count} scenarios")

    print(f"\nTotal scenarios: {final_count}")
    print(f"New scenarios generated: {generated}")

    return final_count


def build_traces_for_new_scenarios():
    """Build traces for newly generated scenarios."""
    print("\n--- Building Traces for New Scenarios ---")

    from experiments.benchmark.trace_builder import TraceBuilder

    scenarios_dir = Path("data/scenarios")
    traces_dir = Path("data/traces")
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Find scenarios without traces
    existing_traces = {f.stem.replace("_trace", "") for f in traces_dir.glob("*_trace.json")}

    builder = TraceBuilder()
    new_traces = 0

    for domain_dir in scenarios_dir.iterdir():
        if not domain_dir.is_dir():
            continue

        for scenario_file in domain_dir.glob("*.json"):
            scenario_id = scenario_file.stem

            if scenario_id in existing_traces:
                continue

            try:
                with open(scenario_file) as f:
                    scenario = json.load(f)

                trace = builder.build_trace(scenario)

                trace_file = traces_dir / f"{scenario_id}_trace.json"
                with open(trace_file, 'w') as f:
                    json.dump(trace, f, indent=2)

                new_traces += 1

                if new_traces % 50 == 0:
                    print(f"  Built {new_traces} new traces...")

            except Exception as e:
                continue

    print(f"  Total new traces built: {new_traces}")
    return new_traces


def create_blind_version_for_new():
    """Create blind versions for new scenarios."""
    print("\n--- Creating Blind Versions ---")

    from experiments.scripts.create_blind_benchmark import (
        blind_trace_file,
        blind_scenario
    )

    scenarios_dir = Path("data/scenarios")
    traces_dir = Path("data/traces")
    blind_dir = Path("data/blind_benchmark")

    # Load existing ground truth
    gt_file = blind_dir / "ground_truth" / "all_ground_truth.json"
    if gt_file.exists():
        with open(gt_file) as f:
            all_gt = json.load(f)
    else:
        all_gt = {}

    new_blind = 0

    for trace_file in traces_dir.glob("*_trace.json"):
        scenario_id = trace_file.stem.replace("_trace", "")

        if scenario_id in all_gt:
            continue

        try:
            with open(trace_file) as f:
                trace = json.load(f)

            # Blind the trace
            blinded, gt = blind_trace_file(trace)

            # Save blinded trace
            blind_trace_file_path = blind_dir / "traces" / trace_file.name
            with open(blind_trace_file_path, 'w') as f:
                json.dump(blinded, f, indent=2)

            all_gt[scenario_id] = gt
            new_blind += 1

        except Exception as e:
            continue

    # Save updated ground truth
    with open(gt_file, 'w') as f:
        json.dump(all_gt, f, indent=2)

    print(f"  Created {new_blind} blind versions")
    print(f"  Total ground truth entries: {len(all_gt)}")


def main():
    """Expand benchmark and build traces."""
    # Expand scenarios
    total = expand_benchmark(target_count=500)

    # Build traces
    build_traces_for_new_scenarios()

    # Create blind versions
    create_blind_version_for_new()

    print("\n" + "="*70)
    print("BENCHMARK EXPANSION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
