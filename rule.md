# Flake8 Error Rules

## Critical Runtime Errors (Showstoppers)

The following Flake8 errors are **showstopper** issues that can halt the runtime with critical errors like `SyntaxError`, `NameError`, etc. These must be fixed immediately as they prevent code execution:

### Error Codes:
- **E901**: SyntaxError or IndentationError
- **E999**: SyntaxError -- failed to compile a file into an Abstract Syntax Tree
- **F821**: Undefined name `name`
- **F822**: Undefined name `name` in `__all__`
- **F823**: Local variable `name` referenced before assignment

These 5 errors are fundamentally different from other Flake8 issues because they represent actual runtime failures rather than style violations.

## Style Violations

All other Flake8 errors are considered "style violations" which are useful for code readability and maintainability but do not affect runtime safety. These can be excluded using:

```xml
<flake8 --exclude>
```

## Usage

When running Flake8, prioritize fixing the showstopper errors (E901, E999, F821, F822, F823) before addressing style violations. Your code will not run properly until these critical errors are resolved.

### Example Configuration

To focus only on critical errors, you can configure Flake8 to ignore all but the showstopper errors:

```ini
[flake8]
# Only check for showstopper errors
select = E901,E999,F821,F822,F823
```

Or to exclude specific style violations while keeping all critical errors:

```ini
[flake8]
# Check all errors but exclude specific style violations
ignore = E203,E266,E501,W503
```
