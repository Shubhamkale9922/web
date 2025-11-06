<?php
// Initialize variables
$result = "";
$num1 = "";
$num2 = "";
$operator = "";

// Check if form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $num1 = isset($_POST['num1']) ? floatval($_POST['num1']) : 0;
    $num2 = isset($_POST['num2']) ? floatval($_POST['num2']) : 0;
    $operator = isset($_POST['operator']) ? $_POST['operator'] : '';

    // Perform calculation based on operator
    switch ($operator) {
        case 'add':
            $result = $num1 + $num2;
            break;
        case 'subtract':
            $result = $num1 - $num2;
            break;
        case 'multiply':
            $result = $num1 * $num2;
            break;
        case 'divide':
            if ($num2 != 0) {
                $result = $num1 / $num2;
            } else {
                $result = "Error: Division by zero!";
            }
            break;
        default:
            $result = "Invalid operator";
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PHP Calculator</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f0f0f0;
            margin: 0;
        }
        .calculator {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        input, select, button {
            margin: 10px;
            padding: 8px;
            font-size: 16px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            font-size: 18px;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="calculator">
        <h2>Simple PHP Calculator</h2>
        <form method="post" action="">
            <input type="number" step="any" name="num1" placeholder="Enter first number" value="<?php echo htmlspecialchars($num1); ?>" required>
            <select name="operator" required>
                <option value="" disabled <?php echo $operator == '' ? 'selected' : ''; ?>>Select Operator</option>
                <option value="add" <?php echo $operator == 'add' ? 'selected' : ''; ?>>+</option>
                <option value="subtract" <?php echo $operator == 'subtract' ? 'selected' : ''; ?>>-</option>
                <option value="multiply" <?php echo $operator == 'multiply' ? 'selected' : ''; ?>>×</option>
                <option value="divide" <?php echo $operator == 'divide' ? 'selected' : ''; ?>>÷</option>
            </select>
            <input type="number" step="any" name="num2" placeholder="Enter second number" value="<?php echo htmlspecialchars($num2); ?>" required>
            <button type="submit">Calculate</button>
        </form>
        <?php if ($result !== ""): ?>
            <div class="result">
                Result: <?php echo htmlspecialchars($result); ?>
            </div>
        <?php endif; ?>
    </div>
</body>
</html>