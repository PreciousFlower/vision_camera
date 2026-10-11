# 手动清理已核对的识别帧缓存；保留原始视频、测试报告、代码和配置。
# 本脚本尚未由 Codex 执行：自动审批检查拒绝了文件删除操作。
$ErrorActionPreference = 'Stop'
$taskWorkspace = 'E:\vibecoding\camera'
$taskCacheRoot = 'E:\vibecoding\camera\phase1\data\frames'
$taskTestRoots = @(
    'E:\vibecoding\camera\phase1\runs\test-video-validation',
    'E:\vibecoding\camera\phase1\runs\_captest'
)

# 清理时网页必须停止，防止采集线程继续生成新帧。
$taskListeners = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 8848,8849,8850 })
if ($taskListeners.Count) {
    throw '网页端口仍有服务监听，请先停止本项目网页服务，再运行清理。'
}

$taskFiles = @()
foreach ($taskRoot in (@($taskCacheRoot) + $taskTestRoots)) {
    if (-not (Test-Path -LiteralPath $taskRoot)) { continue }
    $taskResolvedRoot = (Resolve-Path -LiteralPath $taskRoot).Path
    if ($taskResolvedRoot -ne $taskRoot -or
        -not $taskResolvedRoot.StartsWith($taskWorkspace + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "缓存目录不符合预期：$taskResolvedRoot"
    }
    $taskRootItem = Get-Item -LiteralPath $taskResolvedRoot -Force
    if ($taskRootItem.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "拒绝清理链接目录：$taskResolvedRoot"
    }
    $taskEntries = @(Get-ChildItem -LiteralPath $taskResolvedRoot -Recurse -Force)
    if (@($taskEntries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "缓存目录含链接，已停止：$taskResolvedRoot"
    }
    if ($taskRoot -eq $taskCacheRoot) {
        $taskSelected = @($taskEntries | Where-Object {
            -not $_.PSIsContainer -and ($_.Extension -eq '.jpg' -or $_.Name -eq 'sampling.json')
        })
    } else {
        $taskFrames = @($taskEntries | Where-Object {
            -not $_.PSIsContainer -and $_.Name -match '^f_\d+\.jpg$'
        })
        $taskFrameFolders = @($taskFrames.DirectoryName | Sort-Object -Unique)
        $taskSelected = $taskFrames + @($taskEntries | Where-Object {
            -not $_.PSIsContainer -and $_.Name -eq 'sampling.json' -and
            $_.DirectoryName -in $taskFrameFolders
        })
    }
    foreach ($taskFile in $taskSelected) {
        if (-not $taskFile.FullName.StartsWith($taskResolvedRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "文件不在缓存目录内：$($taskFile.FullName)"
        }
    }
    $taskFiles += $taskSelected
}

$taskFreedBytes = ($taskFiles | Measure-Object Length -Sum).Sum
foreach ($taskFile in $taskFiles) {
    Remove-Item -LiteralPath $taskFile.FullName -Force -ErrorAction Stop
}
Write-Host ('已删除 {0} 个缓存文件，释放约 {1:N1} MB。' -f $taskFiles.Count, ($taskFreedBytes / 1MB))
Write-Host '原始视频和测试报告已保留。空目录会在下一次分析时复用。'
