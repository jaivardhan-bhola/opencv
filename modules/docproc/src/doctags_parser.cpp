// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "doctags_parser.hpp"
#include "text_utils.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace cv { namespace docproc {

namespace {

struct Entry
{
    bool isTag;
    String name;     // valid when isTag; tag name without '<' '>' and without a leading '/'
    bool closing;     // valid when isTag; true for </name>
    String text;      // valid when !isTag; trimmed literal text run
};

std::vector<Entry> tokenize(const String& raw)
{
    std::vector<Entry> entries;
    size_t i = 0, n = raw.size();
    while (i < n)
    {
        size_t lt = raw.find('<', i);
        if (lt == String::npos)
        {
            String text = trimmed(raw.substr(i));
            if (!text.empty())
                entries.push_back({false, String(), false, text});
            break;
        }
        if (lt > i)
        {
            String text = trimmed(raw.substr(i, lt - i));
            if (!text.empty())
                entries.push_back({false, String(), false, text});
        }
        size_t gt = raw.find('>', lt);
        if (gt == String::npos)
            break;
        String inner = raw.substr(lt + 1, gt - lt - 1);
        bool closing = !inner.empty() && inner[0] == '/';
        entries.push_back({true, closing ? inner.substr(1) : inner, closing, String()});
        i = gt + 1;
    }
    return entries;
}

bool isLocTag(const String& name)
{
    return name.rfind("loc_", 0) == 0;
}

bool isHeaderTag(const String& name)
{
    return name.rfind("section_header_level", 0) == 0;
}

bool isTextLikeTag(const String& name)
{
    return isHeaderTag(name) || name == "text" || name == "caption" || name == "page_header" ||
           name == "page_footer" || name == "formula" || name == "picture" || name == "logo" ||
           name == "list_item";
}

bool isTableCellTag(const String& name)
{
    return name == "ched" || name == "fcel" || name == "ecel" || name == "lcel";
}

// Skips the <loc_N> tags immediately following an opening tag, if present; DocTags location
// tags carry no data this parser surfaces (see docproc::Block/Line/Cell::bbox).
void skipLoc(const std::vector<Entry>& entries, size_t& i)
{
    while (i < entries.size() && entries[i].isTag && !entries[i].closing &&
           isLocTag(entries[i].name))
        i++;
}

void parseTable(const std::vector<Entry>& entries, size_t& i, Block& block)
{
    std::vector<std::vector<std::pair<String, String>>> rows;
    std::vector<std::pair<String, String>> curRow;

    while (i < entries.size() &&
           !(entries[i].isTag && entries[i].closing && entries[i].name == "otsl"))
    {
        if (entries[i].isTag && isTableCellTag(entries[i].name))
        {
            String cellType = entries[i].name;
            i++;
            String text;
            if (i < entries.size() && !entries[i].isTag)
            {
                text = entries[i].text;
                i++;
            }
            curRow.push_back({cellType, text});
        }
        else if (entries[i].isTag && entries[i].name == "nl")
        {
            rows.push_back(curRow);
            curRow.clear();
            i++;
        }
        else
        {
            i++;
        }
    }
    if (!curRow.empty())
        rows.push_back(curRow);
    if (i < entries.size())
        i++; // consume </otsl>

    block.rows = (int)rows.size();
    size_t maxCols = 0;
    for (const auto& row : rows)
        maxCols = std::max(maxCols, row.size());
    block.cols = (int)maxCols;
    fillTableCells(block, rows, [](const String& cellType) { return cellType == "ched"; });
}

} // namespace

bool isDocTags(const String& raw)
{
    return raw.find("<doctag>") != String::npos;
}

bool isOtslCellStream(const String& raw)
{
    return raw.find("<fcel>") != String::npos || raw.find("<ched>") != String::npos;
}

void parseOtslCellStream(const String& raw, Page& page)
{
    std::vector<Entry> entries = tokenize(raw);
    size_t i = 0;
    Block block;
    block.type = BLOCK_TABLE;
    block.confidence = 1.f;
    parseTable(entries, i, block); // no </otsl> to stop at; runs to end of the stream
    page.blocks.push_back(block);
}

void parseDocTagsPage(const String& raw, Page& page)
{
    std::vector<Entry> entries = tokenize(raw);
    size_t i = 0, n = entries.size();

    Block textAccum;
    bool haveTextAccum = false;

    auto flushTextAccum = [&]()
    {
        if (haveTextAccum && !textAccum.lines.empty())
            page.blocks.push_back(textAccum);
        textAccum = Block();
        textAccum.type = BLOCK_TEXT;
        textAccum.confidence = 1.f;
        haveTextAccum = false;
    };
    flushTextAccum();

    while (i < n)
    {
        const Entry& e = entries[i];
        if (!e.isTag || e.closing)
        {
            i++;
            continue;
        }

        const String name = e.name;
        if (name == "doctag")
        {
            i++;
            continue;
        }

        if (isTextLikeTag(name))
        {
            i++;
            skipLoc(entries, i);
            String text;
            if (i < n && !entries[i].isTag)
            {
                text = entries[i].text;
                i++;
            }
            if (i < n && entries[i].isTag && entries[i].closing && entries[i].name == name)
                i++;

            if (isHeaderTag(name))
            {
                flushTextAccum();
                Block block;
                block.type = BLOCK_TITLE;
                block.confidence = 1.f;
                Line line;
                line.text = text;
                block.lines.push_back(line);
                page.blocks.push_back(block);
            }
            else
            {
                haveTextAccum = true;
                Line line;
                line.text = text;
                textAccum.lines.push_back(line);
            }
            continue;
        }

        if (name == "otsl")
        {
            flushTextAccum();
            i++;
            skipLoc(entries, i);
            Block block;
            block.type = BLOCK_TABLE;
            block.confidence = 1.f;
            parseTable(entries, i, block);
            page.blocks.push_back(block);
            continue;
        }

        // Unrecognized/wrapper tag (e.g. <unordered_list>): skip it, its matching close is
        // dropped by the "!e.isTag || e.closing" branch above when we reach it.
        i++;
    }

    flushTextAccum();
}

}} // namespace cv::docproc
