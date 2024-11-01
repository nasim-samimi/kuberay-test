import yaml
import os
import copy

# Custom presenter to handle multiline strings in block format
def str_presenter(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

yaml.add_representer(str, str_presenter)  # Register the custom presenter for strings

def modify_resources_and_generate_files(input_yaml_path, output_dir, head_resources, worker_resources, claim_parameters):
    # Load the YAML file as a list of documents
    with open(input_yaml_path, 'r') as f:
        data = list(yaml.safe_load_all(f))  # Convert generator to a list to prevent file closure issues

    # Iterate through resource configurations
    for head_cpu, head_memory in head_resources:
        for worker_cpu, worker_memory in worker_resources:
            for runtime, period in claim_parameters:
                
                # Create a copy for modification
                modified_data = [copy.deepcopy(doc) for doc in data]
                
                # Apply resource configurations
                for doc in modified_data:
                    if doc.get('kind') == 'RayJob':
                        ray_cluster_spec = doc['spec'].get('rayClusterSpec', {})

                        # Modify head resources
                        head_group_spec = ray_cluster_spec.get('headGroupSpec', {})
                        head_template = head_group_spec.get('template', {}).get('spec', {}).get('containers', [])
                        for container in head_template:
                            if container.get('name') == 'ray-head':
                                container['resources']['limits']['cpu'] = head_cpu
                                container['resources']['limits']['memory'] = head_memory
                                container['resources']['requests']['cpu'] = head_cpu
                                container['resources']['requests']['memory'] = head_memory

                        # Modify worker resources
                        worker_group_specs = ray_cluster_spec.get('workerGroupSpecs', [])
                        for worker_group in worker_group_specs:
                            worker_template = worker_group.get('template', {}).get('spec', {}).get('containers', [])
                            for container in worker_template:
                                if container.get('name') == 'ray-worker':
                                    container['resources']['limits']['cpu'] = worker_cpu
                                    container['resources']['limits']['memory'] = worker_memory
                                    container['resources']['requests']['cpu'] = worker_cpu
                                    container['resources']['requests']['memory'] = worker_memory

                        # Explicitly format the `runtimeEnvYAML` and `entrypoint` fields with block notation
                        doc['spec']['runtimeEnvYAML'] = """\
pip:
  - pandas
working_dir: "https://github.com/nasim-samimi/kuberay-test/archive/refs/heads/main.zip"
"""
                        doc['spec']['entrypoint'] = "python mobilenet-imagenet/job.py"

                # Update ResourceClaimTemplate parameters
                for doc in modified_data:
                    if doc.get('kind') == 'RtClaimParameters':
                        doc['spec']['runtime'] = runtime
                        doc['spec']['period'] = period

                # Define a unique filename for each configuration
                output_filename = f"rayjob_head_{head_cpu}_{head_memory}_worker_{worker_cpu}_{worker_memory}_claim_{runtime}_{period}.yaml"
                output_path = os.path.join(output_dir, output_filename)

                # Write the modified data to a new YAML file with all documents
                with open(output_path, 'w') as f:
                    yaml.safe_dump_all(modified_data, f, default_flow_style=False)

# Example usage
input_yaml_path = 'mobilenet-imagenet/job.yaml'
test_name = 'yaml-test'    
output_dir = f'mobilenet-imagenet/{test_name}'
os.makedirs(output_dir, exist_ok=True)

# Lists of resources for head and worker, and claim parameters to iterate over
head_resources = [("1", "1500Mi"), ("2", "2000Mi")]
worker_resources = [("500m", "2Gi"), ("1", "3Gi")]
claim_parameters = [(900, 1000), (1000, 1200)]

modify_resources_and_generate_files(input_yaml_path, output_dir, head_resources, worker_resources, claim_parameters)
