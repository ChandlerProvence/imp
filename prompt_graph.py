#!/usr/bin/env python3
"""
Graph-based prompt generation system.
Supports schema-to-prompt generation, composable prompt nodes, and configurable templates.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
SCHEMAS_DIR = ROOT / "schemas"


def generate_schema_instructions_generic(schema: dict, artifact_type: Optional[str] = None) -> str:
    """
    Fully generic schema-to-prompt generator using recursive traversal.
    Works for any artifact type without hardcoded field handling.
    """
    lines = ["\nARTIFACT JSON SCHEMA REQUIREMENTS:"]
    visited_refs = set()
    
    def resolve_ref(ref: str, schema: dict) -> dict:
        """Resolve $ref references in schema."""
        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            current = schema
            for part in parts:
                current = current.get(part, {})
            return current
        return {}
    
    def get_property_schema(prop: dict, schema: dict) -> dict:
        """Resolve property schema, handling $ref."""
        if "$ref" in prop:
            ref_id = prop["$ref"]
            if ref_id not in visited_refs:
                visited_refs.add(ref_id)
                resolved = resolve_ref(ref_id, schema)
                return resolved
        return prop
    
    def traverse_property(
        path: str,
        prop: dict,
        required: bool,
        schema: dict,
        indent: int = 0,
        parent_description: Optional[str] = None
    ) -> List[str]:
        """Recursively traverse schema properties and generate instructions."""
        result = []
        indent_str = "  " * indent
        
        # Resolve $ref if present
        prop = get_property_schema(prop, schema)
        
        prop_type = prop.get("type")
        description = prop.get("description", parent_description or "")
        enum_values = prop.get("enum")
        pattern = prop.get("pattern")
        const_value = prop.get("const")
        min_items = prop.get("minItems")
        max_items = prop.get("maxItems")
        min_length = prop.get("minLength")
        max_length = prop.get("maxLength")
        
        # Build field description
        field_label = f"{indent_str}- {path}:"
        if required:
            field_label = field_label.replace(":", " (REQUIRED):")
        
        type_info = []
        if const_value:
            type_info.append(f'Exactly "{const_value}"')
        elif enum_values:
            type_info.append(f"One of: {', '.join(f'\"{v}\"' for v in enum_values)}")
        elif prop_type:
            # JSON Schema allows "type": ["string", "integer"] etc.; normalize to string
            type_str = " or ".join(prop_type) if isinstance(prop_type, list) else prop_type
            type_info.append(type_str)
        
        if pattern:
            type_info.append(f"pattern: {pattern}")
        if min_length is not None:
            type_info.append(f"minLength: {min_length}")
        if max_length is not None:
            type_info.append(f"maxLength: {max_length}")
        if min_items is not None:
            type_info.append(f"minItems: {min_items}")
        if max_items is not None:
            type_info.append(f"maxItems: {max_items}")
        
        field_desc = " ".join(type_info)
        if description:
            field_desc += f". {description}"
        
        result.append(f"{field_label} {field_desc}")
        
        # Handle nested structures
        if prop_type == "object":
            required_fields = prop.get("required", [])
            properties = prop.get("properties", {})
            
            if properties:
                result.append(f"{indent_str}  Required fields: {', '.join(required_fields) if required_fields else 'none'}")
                
                for field_name, field_prop in properties.items():
                    field_path = f"{path}.{field_name}" if path else field_name
                    is_required = field_name in required_fields
                    nested = traverse_property(
                        field_path, field_prop, is_required, schema, indent + 1, description
                    )
                    result.extend(nested)
        
        elif prop_type == "array":
            items = prop.get("items", {})
            if items:
                items_schema = get_property_schema(items, schema)
                array_path = f"{path}[]"
                nested = traverse_property(
                    array_path, items_schema, True, schema, indent + 1, description
                )
                result.extend(nested)
        
        return result
    
    # Process top-level required fields
    required_fields = schema.get("required", [])
    properties = schema.get("properties", {})
    
    # Special handling for artifact_type if provided
    if artifact_type and "artifact_type" in properties:
        prop = properties["artifact_type"]
        const_val = prop.get("const")
        if const_val:
            lines.append(f"- artifact_type: Exactly \"{const_val}\" (string).")
        else:
            lines.extend(traverse_property("artifact_type", prop, True, schema))
    
    # Process other required fields
    for field in required_fields:
        if field == "artifact_type" and artifact_type:
            continue  # Already handled
        prop = properties.get(field, {})
        field_lines = traverse_property(field, prop, True, schema)
        lines.extend(field_lines)
    
    # Process definitions (for citation_id, citation, etc.)
    defs = schema.get("definitions", {})
    if defs:
        lines.append("\nSCHEMA DEFINITIONS:")
        for def_name, def_schema in defs.items():
            lines.append(f"\n{def_name.upper().replace('_', ' ')}:")
            def_lines = traverse_property(def_name, def_schema, False, schema, indent=0)
            lines.extend(def_lines)
    
    return "\n".join(lines)


class PromptNode:
    """A single node in the prompt graph."""
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        content: Optional[str] = None,
        template: Optional[str] = None,
        generator: Optional[str] = None,
        config_path: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ):
        self.node_id = node_id
        self.node_type = node_type  # "template", "config_value", "schema_generator", "evidence_formatter", etc.
        self.content = content
        self.template = template
        self.generator = generator
        self.config_path = config_path
        self.dependencies = dependencies or []
        self.output = None
    
    def render(
        self,
        context: Dict[str, Any],
        node_outputs: Dict[str, str]
    ) -> str:
        """Render this node's output given context and dependency outputs."""
        if self.node_type == "template":
            if self.template:
                # Enhanced template substitution with nested access
                rendered = self.template
                
                # Replace dependency outputs first
                for dep_id, dep_output in node_outputs.items():
                    rendered = rendered.replace(f"{{{dep_id}}}", dep_output)
                
                # Replace context values (support nested paths like mission.mission_statement)
                def replace_context(match):
                    key_path = match.group(1)
                    parts = key_path.split(".")
                    value = context
                    for part in parts:
                        if isinstance(value, dict):
                            value = value.get(part)
                        else:
                            value = None
                            break
                        if value is None:
                            break
                    return str(value) if value is not None else f"{{{key_path}}}"
                
                rendered = re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}", replace_context, rendered)
                return rendered
            return self.content or ""
        
        elif self.node_type == "config_value":
            if self.config_path:
                # Navigate config path (e.g., "task_config.description")
                parts = self.config_path.split(".")
                value = context
                for part in parts:
                    if isinstance(value, dict):
                        value = value.get(part, {})
                    else:
                        value = None
                        break
                    if value is None:
                        break
                
                # Handle special cases
                if isinstance(value, dict) and self.config_path == "task_config.output_overrides":
                    max_coas = value.get("max_coas", 3)
                    exec_bullets = value.get("executive_summary_bullets", 3)
                    return f"\nOUTPUT CONSTRAINTS:\n- Maximum COAs: {max_coas}\n- Executive summary bullets: {exec_bullets}\n"
                
                return str(value) if value is not None else ""
            return self.content or ""
        
        elif self.node_type == "schema_generator":
            schema = context.get("schema", {})
            artifact_type = context.get("artifact_type")
            return generate_schema_instructions_generic(schema, artifact_type)
        
        elif self.node_type == "evidence_formatter":
            evidence = context.get("evidence", {})
            formatted, _ = format_evidence(evidence)
            return f"EVIDENCE (USE ONLY THIS INFORMATION):\n{formatted}"
        
        elif self.node_type == "static":
            return self.content or ""
        
        return ""


class PromptGraph:
    """Graph-based prompt composition system."""
    
    def __init__(self):
        self.nodes: Dict[str, PromptNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (from_id, to_id)
        self.execution_order: List[str] = []
    
    def add_node(self, node: PromptNode):
        """Add a node to the graph."""
        self.nodes[node.node_id] = node
    
    def add_edge(self, from_id: str, to_id: str):
        """Add a dependency edge."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError(f"Nodes {from_id} or {to_id} not found")
        self.edges.append((from_id, to_id))
        if from_id not in self.nodes[to_id].dependencies:
            self.nodes[to_id].dependencies.append(from_id)
    
    def topological_sort(self) -> List[str]:
        """Determine execution order using topological sort."""
        in_degree = defaultdict(int)
        for node_id in self.nodes:
            in_degree[node_id] = 0
        
        for from_id, to_id in self.edges:
            in_degree[to_id] += 1
        
        queue = [node_id for node_id in self.nodes if in_degree[node_id] == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            for from_id, to_id in self.edges:
                if from_id == node_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")
        
        return result
    
    def build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt by executing nodes in dependency order."""
        execution_order = self.topological_sort()
        node_outputs = {}
        
        for node_id in execution_order:
            node = self.nodes[node_id]
            output = node.render(context, node_outputs)
            node_outputs[node_id] = output
        
        # Concatenate outputs in execution order
        return "\n\n".join(node_outputs[node_id] for node_id in execution_order if node_outputs[node_id])


def load_prompt_graph_from_config(graph_config: dict) -> PromptGraph:
    """Load prompt graph from YAML configuration."""
    graph = PromptGraph()
    
    nodes_config = graph_config.get("nodes", [])
    for node_cfg in nodes_config:
        node = PromptNode(
            node_id=node_cfg["id"],
            node_type=node_cfg["type"],
            content=node_cfg.get("content"),
            template=node_cfg.get("template"),
            generator=node_cfg.get("generator"),
            config_path=node_cfg.get("config_path"),
            dependencies=node_cfg.get("dependencies", [])
        )
        graph.add_node(node)
    
    edges_config = graph_config.get("edges", [])
    for edge_cfg in edges_config:
        graph.add_edge(edge_cfg["from"], edge_cfg["to"])
    
    return graph


def format_evidence(evidence_data: dict) -> Tuple[str, List[dict]]:
    """Format evidence from fixture for prompt. Returns (formatted_text, citation_map)."""
    results = evidence_data.get("results", [])
    blocks = []
    citation_map = []
    for i, result in enumerate(results, 1):
        cit_id = f"CIT-{i:04d}"
        title = result.get("title", "Untitled")
        snippet = result.get("snippet", "")
        citation = result.get("citation_handle", {})
        source = citation.get("source_system", "unknown")
        doc_id = citation.get("document_id", "unknown")
        blocks.append(f"EVIDENCE {i} (use citation_id {cit_id}):\nSource: {source} (Document: {doc_id})\nTitle: {title}\nContent: {snippet}")
        citation_map.append({"citation_id": cit_id, "source_system": source, "document_id": doc_id, "snippet": snippet[:1200]})
    return "\n\n".join(blocks) if blocks else "No evidence available.", citation_map


def load_prompt_template(template_path: str) -> dict:
    """Load prompt template from YAML file."""
    import yaml
    path = Path(template_path)
    if not path.is_absolute():
        path = ROOT / "configs" / "prompt_templates" / template_path
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt_from_graph(
    task_config: dict,
    mission: dict,
    purpose_config: dict,
    evidence: dict,
    schema: dict
) -> str:
    """
    Build prompt using graph-based composition.
    Falls back to task_config.prompt_graph if present, otherwise uses default graph.
    """
    # Check if task has custom prompt graph config or template reference
    prompt_graph_cfg = task_config.get("prompt_graph")
    prompt_template_ref = task_config.get("prompt_template")
    
    if prompt_template_ref:
        # Load template from file
        template_cfg = load_prompt_template(prompt_template_ref)
        graph = load_prompt_graph_from_config(template_cfg)
    elif prompt_graph_cfg:
        # Load custom prompt graph from config
        graph = load_prompt_graph_from_config(prompt_graph_cfg)
    else:
        # Use default graph based on artifact type
        graph = _build_default_prompt_graph(task_config, mission, purpose_config)
    
    # Resolve purpose config values
    def resolve_purpose_value(cfg_key: str, mission_key: str, default: str) -> str:
        val = purpose_config.get(cfg_key)
        if isinstance(val, dict):
            key = mission.get(mission_key) or purpose_config.get(f"default_{cfg_key}")
            return (val.get(key) if key else "") or key or default
        return val or default
    
    audience_cfg = purpose_config.get("audience")
    if isinstance(audience_cfg, dict):
        selected_audience = mission.get("audience") or purpose_config.get("default_audience") or "executive"
        purpose_description = (audience_cfg.get(selected_audience) or purpose_config.get("description") or "").strip()
        audience = selected_audience
    else:
        audience = audience_cfg or "general"
        purpose_description = purpose_config.get("description", "")
    
    intended_use = resolve_purpose_value("intended_use", "intended_use_key", "planning")
    release_level = resolve_purpose_value("release_level", "release_level_key", "draft")
    time_sensitivity = resolve_purpose_value("time_sensitivity", "time_sensitivity_key", "normal")
    risk_tolerance = resolve_purpose_value("risk_tolerance", "risk_tolerance_key", "moderate")
    format_preference = purpose_config.get("format_preference", "standard")
    
    output_overrides = task_config.get("output_overrides", {}) or {}
    max_coas = output_overrides.get("max_coas", 3)
    exec_bullets = output_overrides.get("executive_summary_bullets", 3)
    
    # Build context for graph execution
    context = {
        "mission": mission,
        "task_config": task_config,
        "purpose_config": purpose_config,
        "evidence": evidence,
        "schema": schema,
        "artifact_type": task_config.get("artifact_type", ""),
        "mission_type": mission.get("mission_type", ""),
        "task_id": mission.get("task_id", ""),
        "mission_statement": mission.get("mission_statement", ""),
        "aoi": mission.get("aoi") or mission.get("aor") or "AOI-UNSPECIFIED",
        "time_window": mission.get("time_window", "UNSPECIFIED"),
        "organization_id": mission.get("organization_id", ""),
        "team_id": mission.get("team_id", ""),
        "audience": audience,
        "purpose_description": purpose_description,
        "intended_use": intended_use,
        "release_level": release_level,
        "time_sensitivity": time_sensitivity,
        "risk_tolerance": risk_tolerance,
        "format_preference": format_preference,
        "max_coas": max_coas,
        "exec_bullets": exec_bullets,
    }
    
    return graph.build_prompt(context)


def _build_default_prompt_graph(task_config: dict, mission: dict, purpose_config: dict) -> PromptGraph:
    """Build default prompt graph for tasks without custom prompt configuration."""
    graph = PromptGraph()
    
    # System role node
    graph.add_node(PromptNode(
        "system_role",
        "template",
        template="You are supporting a {mission_type} analyst performing task: {task_id} (artifact type: {artifact_type})."
    ))
    
    # Requirements node
    graph.add_node(PromptNode(
        "requirements",
        "static",
        content="""CRITICAL REQUIREMENTS - NO HALLUCINATION:
1. EVERY claim, fact, assessment, or finding MUST be directly supported by the provided evidence
2. DO NOT invent, assume, or infer information not explicitly stated in the evidence
3. DO NOT use general knowledge or training data - ONLY use the provided evidence
4. If evidence is insufficient for a required field, state "Insufficient evidence" rather than inventing content
5. EVERY statement must be traceable to a specific evidence source
6. Prioritize evidence from preferred/trusted sources when available
7. If you cannot substantiate a claim with evidence, explicitly state the limitation"""
    ))
    
    # Task description node
    graph.add_node(PromptNode(
        "task_description",
        "config_value",
        config_path="task_config.description"
    ))
    
    # Mission context node
    graph.add_node(PromptNode(
        "mission_context",
        "template",
        template="""MISSION CONTEXT:
Mission Statement: {mission_statement}
AOI/AOR: {aoi}
Time Window: {time_window}
Organization: {organization_id}
Team: {team_id}"""
    ))
    
    # Purpose profile node
    graph.add_node(PromptNode(
        "purpose_profile",
        "template",
        template="""PURPOSE PROFILE (WHY — SCOPE YOUR OUTPUT ACCORDINGLY):
Audience: {audience}
Intended Use: {intended_use}
Release Level: {release_level}
Time Sensitivity: {time_sensitivity}
Risk Tolerance: {risk_tolerance}"""
    ))
    
    # Evidence node
    graph.add_node(PromptNode(
        "evidence",
        "evidence_formatter"
    ))
    
    # Chain-of-thought node
    graph.add_node(PromptNode(
        "cot_instructions",
        "static",
        content="""APPROACH THIS TASK USING CHAIN-OF-THOUGHT REASONING. Show your reasoning process step-by-step, then provide the final artifact.

STEP 1: EVIDENCE ANALYSIS
- Review each evidence block systematically
- Identify key facts, claims, and data points from each source
- Note source credibility and recency
- Identify any conflicts or gaps in evidence
- List specific evidence references for each key finding

STEP 2: PATTERN IDENTIFICATION
- Look for patterns across evidence sources
- Identify recurring themes or trends
- Note temporal patterns if timestamps available
- Identify correlations between different evidence points

STEP 3: SYNTHESIS AND INTERPRETATION
- Synthesize evidence into coherent insights
- Connect evidence points to mission objectives
- Identify implications for decision-making
- Note limitations and uncertainties
- Ensure every insight traces back to specific evidence

STEP 4: ARTIFACT CONSTRUCTION
- Structure your findings according to the required artifact format
- Ensure every claim has an evidence citation
- Prioritize evidence from preferred sources when marked
- Acknowledge gaps explicitly rather than filling with assumptions"""
    ))
    
    # Schema instructions node
    graph.add_node(PromptNode(
        "schema_instructions",
        "schema_generator"
    ))
    
    # Build edges (dependency graph)
    graph.add_edge("system_role", "requirements")
    graph.add_edge("requirements", "task_description")
    graph.add_edge("task_description", "mission_context")
    graph.add_edge("mission_context", "purpose_profile")
    graph.add_edge("purpose_profile", "evidence")
    graph.add_edge("evidence", "cot_instructions")
    graph.add_edge("cot_instructions", "schema_instructions")
    
    return graph
