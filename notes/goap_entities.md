## GOAP Entity Parsing

### Introduction
The GOAP (Goal-Oriented Action Planning) Entity Parsing framework is designed to parse and structure arbitrary sentences into a logical, object-oriented format. This framework is essential for understanding complex user inputs and transforming them into structured data models.

### Key Concepts and Entities

#### Entity and Attribute
- **Entity**: A primary unit representing an object or concept, characterized by attributes.
- **Attribute**: Characteristics or properties of an entity.

#### Statement
- **Description**: A logical construct that defines conditions or facts about an entity's attributes, including a boolean condition.

#### Composite Statement
- **Role**: Combines multiple statements using logical operators (AND, OR, NOT) to form more complex conditions.

#### Proposition
- **Function**: Asserts composite statements that are true for specific entities.

#### World State
- **Definition**: A representation of the current state of the system, constituted by simultaneously true propositions about all entities.

#### Affordance
- **Purpose**: Defines potential actions involving entities, based on requisites and resulting in consequences for source and target entities.

#### Action
- **Usage**: A specific instance of an affordance being applied from a source entity to a target.

### Process Flow

1. **Parsing and Structuring**: Breaks down sentences into entities and attributes, formulating statements and composite statements.
2. **Logical Interpretation**: Uses propositions to assert truths about entities based on the parsed statements.
3. **World State Construction**: Constructs the world state by aggregating all current propositions about entities.
4. **Affordance and Action Identification**: Identifies possible actions (affordances) based on the world state and entities involved.

### Application
This parsing framework can be applied to various contexts, from natural language processing in AI conversational agents to complex decision-making systems, providing a foundation for logical reasoning and planning.

