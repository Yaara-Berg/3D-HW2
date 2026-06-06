local function header_text(cell)
  return pandoc.utils.stringify(cell.contents):lower()
end

local function ncol(tbl)
  return #tbl.head.rows[1].cells
end

local function widths_match(tbl, widths)
  return ncol(tbl) == #widths
end

local function apply_widths(tbl, widths, aligns)
  local specs = {}
  for i, w in ipairs(widths) do
    local align = aligns[i] or pandoc.AlignDefault
    specs[i] = { align, w }
  end
  tbl.colspecs = specs
  return tbl
end

local table_configs = {
  {
    match = function(tbl)
      if #tbl.head.rows < 1 or ncol(tbl) ~= 7 then return false end
      local r = tbl.head.rows[1].cells
      return header_text(r[1]) == "run"
        and header_text(r[6]) == "notes"
        and header_text(r[7]):match("^psnr")
    end,
    widths = { 0.06, 0.08, 0.10, 0.10, 0.10, 0.46, 0.10 },
    aligns = {
      pandoc.AlignCenter, pandoc.AlignCenter, pandoc.AlignCenter,
      pandoc.AlignCenter, pandoc.AlignCenter, pandoc.AlignLeft, pandoc.AlignRight,
    },
  },
  {
    match = function(tbl)
      if #tbl.head.rows < 1 or ncol(tbl) ~= 8 then return false end
      local r = tbl.head.rows[1].cells
      return header_text(r[1]) == "run"
        and header_text(r[7]) == "notes"
        and (header_text(r[3]):match("omega") or header_text(r[3]):match("ω"))
    end,
    widths = { 0.05, 0.06, 0.11, 0.11, 0.09, 0.10, 0.38, 0.10 },
    aligns = {
      pandoc.AlignCenter, pandoc.AlignCenter, pandoc.AlignCenter, pandoc.AlignCenter,
      pandoc.AlignCenter, pandoc.AlignCenter, pandoc.AlignLeft, pandoc.AlignRight,
    },
  },
  {
    match = function(tbl)
      if #tbl.head.rows < 1 or ncol(tbl) ~= 2 then return false end
      local r = tbl.head.rows[1].cells
      return header_text(r[1]) == "setting" and header_text(r[2]) == "value"
    end,
    widths = { 0.38, 0.62 },
    aligns = { pandoc.AlignLeft, pandoc.AlignLeft },
  },
}

function Table(tbl)
  for _, cfg in ipairs(table_configs) do
    if cfg.match(tbl) and widths_match(tbl, cfg.widths) then
      return apply_widths(tbl, cfg.widths, cfg.aligns or {})
    end
  end
  return tbl
end
