import re
from copy import deepcopy
from typing import Optional, List, Dict, Set, Tuple, Any


class PromptTemplate:
    def __init__(self, template: str, input_variables: Optional[List[str]] = None):
        self.template = template
        self._input_variables = input_variables
        self._positional_index = 0
        self._filled_positional = {}
        self._filled_keywords = {}

        if input_variables is None:
            self._input_variables = list(self._extract_variables())

    def _extract_variables(self) -> Set[str]:
        pattern = r"(?<!\{)\{([^{}]+)\}(?!\})"
        matches = re.findall(pattern, self.template)
        variables = set()
        for match in matches:
            match = match.strip()
            if match:
                variables.add(match)
        return variables

    def _unescape_braces(self, text: str) -> str:
        text = text.replace("{{", "\x00")
        text = text.replace("}}", "\x01")
        return text

    def _escape_braces(self, text: str) -> str:
        text = text.replace("\x00", "{")
        text = text.replace("\x01", "}")
        return text

    def _render_template(
        self, template: str, positional: Dict[int, Any], keywords: Dict[str, Any]
    ) -> str:
        temp = self._unescape_braces(template)

        for idx, value in positional.items():
            placeholder = "{" + str(idx) + "}"
            temp = temp.replace(placeholder, str(value))

        for key, value in keywords.items():
            placeholder = "{" + key + "}"
            temp = temp.replace(placeholder, str(value))

        return self._escape_braces(temp)

    def format(self, **kwargs) -> "PromptTemplate":
        new_template = PromptTemplate(self.template, self._input_variables)
        new_template._positional_index = self._positional_index
        new_template._filled_positional = deepcopy(self._filled_positional)
        new_template._filled_keywords = deepcopy(self._filled_keywords)

        new_template._filled_keywords.update(kwargs)

        return new_template

    def _parse_lshift_args(self, args) -> Tuple[List[Any], Dict[str, Any]]:
        positional = []
        keywords = {}
        found_dict = False

        if isinstance(args, dict):
            return [], args

        if not isinstance(args, (tuple, list)):
            return [args], {}

        for item in args:
            if isinstance(item, dict):
                found_dict = True
                keywords.update(item)
            else:
                if found_dict:
                    raise ValueError(
                        "Positional arguments must come before keyword arguments"
                    )
                positional.append(item)

        return positional, keywords

    def __lshift__(self, args) -> "PromptTemplate":
        positional, keywords = self._parse_lshift_args(args)

        new_template = PromptTemplate(self.template, self._input_variables)
        new_template._positional_index = self._positional_index
        new_template._filled_positional = deepcopy(self._filled_positional)
        new_template._filled_keywords = deepcopy(self._filled_keywords)

        for value in positional:
            new_template._filled_positional[new_template._positional_index] = value
            new_template._positional_index += 1

        new_template._filled_keywords.update(keywords)

        return new_template

    def __add__(self, other: "PromptTemplate") -> "PromptTemplate":
        if not isinstance(other, PromptTemplate):
            raise TypeError("Can only concatenate with another PromptTemplate")

        combined_template = self.template + other.template
        new_prompt = PromptTemplate(combined_template)

        new_prompt._filled_positional = deepcopy(self._filled_positional)
        new_prompt._filled_keywords = deepcopy(self._filled_keywords)

        offset = self._positional_index
        for idx, value in other._filled_positional.items():
            new_prompt._filled_positional[idx + offset] = value

        new_prompt._filled_keywords.update(other._filled_keywords)
        new_prompt._positional_index = self._positional_index + other._positional_index

        return new_prompt

    def __or__(self, other: "PromptTemplate") -> "PromptTemplate":
        if not isinstance(other, PromptTemplate):
            raise TypeError("Can only pipe with another PromptTemplate")

        combined_template = self.template + " | " + other.template
        new_prompt = PromptTemplate(combined_template)

        new_prompt._filled_positional = deepcopy(self._filled_positional)
        new_prompt._filled_keywords = deepcopy(self._filled_keywords)

        offset = self._positional_index
        for idx, value in other._filled_positional.items():
            new_prompt._filled_positional[idx + offset] = value

        new_prompt._filled_keywords.update(other._filled_keywords)
        new_prompt._positional_index = self._positional_index + other._positional_index

        return new_prompt

    def totext(self) -> str:
        return self._render_template(
            self.template, self._filled_positional, self._filled_keywords
        )

    def get_remaining_variables(self) -> Set[str]:
        all_vars = self._extract_variables()
        filled_vars = set(self._filled_keywords.keys())
        filled_vars.update(str(idx) for idx in self._filled_positional.keys())
        return all_vars - filled_vars

    def __repr__(self) -> str:
        return f"PromptTemplate(template={self.template!r})"

    def __str__(self) -> str:
        return self.totext()


if __name__ == "__main__":

    prompt1 = PromptTemplate("Hello {name}, you are {age} years old")
    result1 = prompt1.format(name="Alice")
    print(f"Result: {result1.totext()}")
    print(f"Result: {result1.format(age=25).totext()}")
    print()

    prompt2 = PromptTemplate("Hello {name}, you are {age} years old, from {city}")
    step1 = prompt2.format(name="Bob")
    step2 = step1.format(age=30)
    step3 = step2.format(city="Shanghai")
    print(f"Result: {step3.totext()}")
    print()
