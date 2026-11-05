param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$ScriptDir\port_model.py" @ArgsList
