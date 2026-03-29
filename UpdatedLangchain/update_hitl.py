import json

notebook_path = "d:/WEB-DEV/GenAI/Langchain-crash-course/UpdatedLangchain/6.Middleware.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        
        # 1. Update the agent creation cell to include a system prompt
        has_agent_creation = any("agent = create_react_agent(" in line for line in source)
        if has_agent_creation and any("execute_database_drop" in line for line in source):
            new_source = []
            for line in source:
                if 'interrupt_before=["tools"]' in line and not line.endswith(','):
                    new_source.append(line.replace('interrupt_before=["tools"] # Pauses BEFORE executing the tools node', 'interrupt_before=["tools"], # Pauses BEFORE executing the tools node'))
                elif ')' in line and line.strip() == ')':
                    # Insert the state_modifier BEFORE the closing paren
                    if 'state_modifier' not in ''.join(source):
                        new_source.append('    state_modifier="You are an admin robot. When the user asks to drop the database, you MUST immediately call the execute_database_drop tool without asking for confirmation."\n')
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

        # 2. Add some print safety to the rejection cell to make it clearer
        has_reject = any("config_reject =" in line for line in source)
        if has_reject:
            new_source = []
            for line in source:
                if 'tool_call = last_message.tool_calls[0]' in line:
                    new_source.extend([
                        "if not last_message.tool_calls:\n",
                        "    print('ERROR: The LLM refused to call the tool! It responded:', last_message.content)\n",
                        "else:\n",
                        "    tool_call = last_message.tool_calls[0]\n",
                        "    \n",
                        "    rejection_message = ToolMessage(\n",
                        "        tool_call_id=tool_call[\"id\"],\n",
                        "        name=tool_call[\"name\"],\n",
                        "        content=\"User REJECTED this action. Tell them that the database drop was aborted.\"\n",
                        "    )\n",
                        "    \n",
                        "    # Update the state with the rejection (as if the tool returned an error/denial)\n",
                        "    agent.update_state(config_reject, {\"messages\": [rejection_message]})\n",
                        "    \n",
                        "    # Resume the agent with the injected rejection message\n",
                        "    rejected_response = agent.invoke(None, config_reject)\n",
                        "    print(\"Final result after rejection:\", rejected_response[\"messages\"][-1].content)\n"
                    ])
                    break # Replaced the rest of the cell manually
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
