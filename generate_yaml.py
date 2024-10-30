import yaml
import os

def modify_resources_and_generate_files(input_yaml_path, output_dir, head_resources, worker_resources, claim_parameters):
    # Load the YAML file
    with open(input_yaml_path, 'r') as f:
        data = list(yaml.safe_load_all(f))  # Load all documents in the YAML file

    # Iterate through resource and claim parameter options to create multiple files
    for head_cpu, head_memory in head_resources:
        for worker_cpu, worker_memory in worker_resources:
            for runtime, period in claim_parameters:
                
                # Create a deep copy of the original data to modify for each combination
                modified_data = [doc.copy() for doc in data]
                
                # Update RayJob resources
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
                
                # Update ResourceClaimTemplate parameters
                for doc in modified_data:
                    if doc.get('kind') == 'RtClaimParameters':
                        doc['spec']['runtime'] = runtime
                        doc['spec']['period'] = period

                # Define a unique filename for each configuration
                output_filename = f"rayjob_head_{head_cpu}_{head_memory}_worker_{worker_cpu}_{worker_memory}_claim_{runtime}_{period}.yaml"
                output_path = os.path.join(output_dir, output_filename)

                # Write the modified data to a new YAML file
                with open(output_path, 'w') as f:
                    yaml.dump_all(modified_data, f, default_flow_style=False)

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
