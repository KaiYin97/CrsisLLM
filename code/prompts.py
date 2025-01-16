from enum import Enum
from typing import List 
from dataclasses import dataclass

class AgentType(Enum):
    USER = "User"
    SYSTEM = "System"

@dataclass
class Category:
    name: str
    description: str  

@dataclass
class ConversationTurn:
    message: str 

SYSTEM_MESSAGE = "You are an excellent AI assistant specialized in processing and analyzing multi-turn conversations related to natural disasters."

PROMPT_TASK1 = "Task: For the below User message, check which event category it belongs to with the below EVENT CATEGORIES given the DESCRIPTION of each EVENT CATEGORY."
PROMPT_TASK2 = "Task: For the above User message, check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY."
PROMPT_TASK3 = "Task: For the above User message, find all location entities contained in it."
PROMPT_TASK4 = """Task: For the above User message, first check which event category it belongs to with the below EVENT CATEGORIES given the DESCRIPTION of each EVENT CATEGORY; 
second, check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY; 
finally find all location entities contained in it."""

PROMPT_TASK3_BENCH = "Task: For the above User message, check whether it is INFORMATIVE for human aid."
PROMPT_TASK2_BENCH = "Task: For the above User message, check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY.."
PROMPT_TASK4_BENCH = """Task: For the above User message, first check which event category it belongs to with the below EVENT CATEGORIES given the DESCRIPTION of each EVENT CATEGORY; 
second, check whether it is INFORMATIVE for human aid.
finally check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY"""

PROMPT_TASK4_INFERENCE = """Task: For the below User message, first check which event category it belongs to with the below EVENT CATEGORIES given the DESCRIPTION of each EVENT CATEGORY; 
second, check which humanitarian aid category it belongs to with the below HUMANITARIAN AID CATEGORIES given the DESCRIPTION of each HUMANITARIAN AID CATEGORY; 
finally find all location entities contained in it. Just output the answer. No Explanation needed"""

EVENT_CATEGORIES = """
<BEGIN EVENT CATEGORIES>
$event_categories
<END EVENT CATEGORIES>
"""

EVENT_CATEGORIES_DESCRP = """
<BEGIN EVENT CATEGORIES DESCRIPTION>
$event_categories
<END EVENT CATEGORIES DESCRIPTION>
"""

HUMAN_CATEGORIES = """
<BEGIN HUMANITARIAN AID CATEGORIES>
$human_categories
<END HUMANITARIAN AID CATEGORIES>
"""

HUMAN_CATEGORIES_DESCRP = """
<BEGIN HUMANITARIAN AID CATEGORIES DESCRIPTION>
$human_categories
<END HUMANITARIAN AID CATEGORIES DESCRIPTION>
"""


USER_MESSAGE = """
<BEGIN USER MESSAGE>
$message
<END USER MESSAGE>
"""

PROMPT_INSTRUCTIONS_TASK1 = """
You should provide your response in JSON format, with the following structure: {"EVENT_CATEGORY": event category the User Message belongs to}
"""

PROMPT_INSTRUCTIONS_TASK2 = """
You should provide your response in JSON format, with the following structure: {"HUMANITARIAN_AID": humanitarian aid category the User Message belongs to}
"""

PROMPT_INSTRUCTIONS_TASK3 = """
You should provide your response in JSON format with all location entities in a List object, with the following structure: {"LOCATIONS": List of location entities the User Message contained}.
"""

PROMPT_INSTRUCTIONS_TASK4 = """
You should provide your response in JSON format, with the following structure: 
{"EVENT_CATEGORY": event category the User Message belongs to, "HUMANITARIAN_AID": humanitarian aid category the User Message belongs to, "LOCATIONS": List of location entities the User Message contained}
"""


PROMPT_INSTRUCTIONS_TASK3_BENCH = """
You should provide your response in JSON format, with the following structure: {"INFORMATIVE": answer "informative" if it is useful for human aid, otherwise answer "not_informative"}
"""
PROMPT_INSTRUCTIONS_TASK2_BENCH = """
You should provide your response in JSON format, with the following structure: {"HUMANITARIAN_AID": humanitarian aid category the User Message belongs to}
"""
PROMPT_INSTRUCTIONS_TASK4_BENCH = """
You should provide your response in JSON format, with the following structure: 
{"EVENT_CATEGORY": event category the User Message belongs to, "INFORMATIVE": answer "informative" if it is useful for human aid, otherwise answer "not_informative", "HUMANITARIAN_AID": humanitarian aid category the User Message belongs to}
"""

Event_category = [
    Category(
        "wildfires",
        "Wildfires: Uncontrolled fires that spread rapidly through vegetation, often caused by natural or human factors."
    ),
    Category(
        "cyclone",
        "Cyclone: a large-scale air mass that rotates around a strong center of low atmospheric pressure, typically leading to strong winds and heavy rain."
    ),
    Category(
        "earthquake",
        "Earthquake: sudden shaking of the ground caused by the movement of tectonic plates beneath the Earth's surface"
    ),
    Category(
        "hurricane",
        "Hurricane: a powerful tropical storm characterized by strong winds, heavy rain, and potential for severe coastal damage."
    ),
    Category(
        "floods",
        "Floods: the overflow of water onto normally dry land, often caused by excessive rainfall, river overflow, or storm surges."
    )]

Event_category_bench = [
    Category(
        "earthquake",
        "Earthquake: a sudden shaking of the ground caused by the movement of tectonic plates beneath the Earth's surface"
    ),
    Category(
        "hurricane",
        "Hurricane: a powerful tropical storm characterized by strong winds, heavy rain, and potential for severe coastal damage."
    ),
    Category(
        "flood",
        "Floods: the overflow of water onto normally dry land, often caused by excessive rainfall, river overflow, or storm surges."
    ),
    Category(
        "bombing",
        "Bombing: An intentional explosion caused by a bomb, often resulting in significant destruction and casualties."
    ),
    Category(
        "collapse",
        "Collapse: The sudden failure or falling down of a structure, often leading to damage and potential loss of life."
    ),
    Category(
        "crash",
        "Crash: An accidental collision involving vehicles, typically resulting in damage and sometimes injuries or fatalities."
    ),
    Category(
        "disaster_events",
        "Disaster Events: A broad category encompassing natural or man-made catastrophes that cause widespread damage, disruption, and loss."
    ),
    Category(
        "disease",
        "Disease: A pathological condition affecting a living organism, often leading to health deterioration and potentially causing epidemics or pandemics."
    ),
    Category(
        "explosion",
        "Explosion: A violent release of energy causing a rapid expansion of gases, typically resulting in a loud noise and significant damage."
    ),
    Category(
        "fire",
        "Fire: The rapid oxidation of material in the chemical process of combustion, producing heat, light, and often causing destruction."
    ),
    Category(
        "hazard",
        "Hazard: A potential source of harm or adverse effects on health, property, or the environment."
    ),
    Category(
        "shooting",
        "Shooting: An act of discharging a firearm, often resulting in injury or death."
    ),
    Category(
        "volcano",
        "Volcano: A rupture in the Earth's crust where molten lava, ash, and gases erupt, often causing widespread environmental and structural damage."
    ),
    Category(
        "landslide",
        "Hurricane: A large and powerful tropical storm system with high winds and heavy rains, capable of causing widespread destruction."
    )
]

Hum_aid_category = [
    Category(
        "caution_and_advice",
        "caution_and_advice: Guidance or warnings provided to the public to help them stay safe or prepare for a disaster."
    ),
    Category(
        "displaced_people_and_evacuations",
        "displaced_people_and_evacuations: Information about people who have been forced to leave their homes due to a disaster and efforts to relocate them."
    ),
    Category(
        "infrastructure_and_utility_damage",
        "infrastructure_and_utility_damage: Reports on the destruction or impairment of buildings, roads, power lines, and other essential services due to a disaster."
    ),
    Category(
        "injured_or_dead_people",
        "injured_or_dead_people: Information regarding casualties, including the number of people injured or killed due to a disaster."
    ),
    Category(
        "missing_or_found_people",
        "missing_or_found_people: Reports on individuals who are unaccounted for or have been located after being reported missing during a disaster."
    ),
    Category(
        "requests_or_urgent_needs",
        "requests_or_urgent_needs: Appeals for immediate assistance, resources, or support to address critical situations during or after a disaster."
    ),
    Category(
        "rescue_volunteering_or_donation_effort",
        "rescue_volunteering_or_donation_effort: Information about rescue operations, volunteer involvement, and efforts to collect donations to support disaster-affected areas."
    )
]

Hum_aid_category_bench = [
    Category(
        "caution_and_advice",
        "caution_and_advice: Guidance or warnings provided to the public to help them stay safe or prepare for a disaster."
    ),
    Category(
        "displaced_and_evacuations",
        "displaced_and_evacuations: Information about people who have been forced to leave their homes due to a disaster and efforts to relocate them."
    ),
    Category(
        'infrastructure_and_utilities_damage',
        "infrastructure_and_utilities_damage: Reports on the destruction or impairment of buildings, roads, power lines, and other essential services due to a disaster."
    ),
    Category(
        "injured_or_dead_people",
        "injured_or_dead_people: Information regarding casualties, including the number of people injured or killed due to a disaster."
    ),
    Category(
        'missing_and_found_people',
        "missing_and_found_people: Reports on individuals who are unaccounted for or have been located after being reported missing during a disaster."
    ),
    Category(
        'requests_or_needs',
        "requests_or_needs: Appeals for immediate assistance, resources, or support to address critical situations during or after a disaster."
    ),
    Category(
        'affected_individual',
        "affected_individual: A person who has been directly impacted by a disaster or emergency situation, requiring assistance or support."
    ),
    Category(
        'disease_related',
        "disease_related: Information or updates pertaining to the outbreak, spread, or impact of diseases, often in the context of public health emergencies."
    ),
    Category(
        'donation_and_volunteering',
        "donation_and_volunteering: Efforts to gather resources, funds, or manpower to support relief operations or assist those affected by disasters."
    ),
    Category(
        'not_humanitarian',
        "not_humanitarian: Information or activities that do not relate directly to humanitarian aid or relief efforts."
    ),
    Category(
        'other_relevant_information',
        "other_relevant_information: Additional details or data that are pertinent to understanding or responding to a disaster, but do not fit into other specific categories."
    ),
    Category(
        'personal_update',
        "personal_update: Information shared by individuals about their personal safety, status, or experiences during a disaster or emergency."
    ),
    Category(
        'physical_landslide',
        "physical_landslide: A sudden and massive movement of earth, rocks, or debris down a slope, often causing significant damage and requiring immediate response."
    ),
    Category(
        'response_efforts',
        "response_efforts: Actions taken by organizations, governments, or individuals to mitigate the impact of a disaster and provide relief to those affected."
    ),
    Category(
        'sympathy_and_support',
        "sympathy_and_support: Expressions of concern, condolences, or encouragement offered to those affected by a disaster or emergency."
    ),
    Category(
        'terrorism_related',
        "terrorism_related: Information or updates related to acts of terrorism, including the response and impact on affected populations."
    )
]

SYSTEM_PREFIX = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
USER_PREFIX = "<|start_header_id|>user<|end_header_id|>"
ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>"

IGNORE_IDX = -100